import os
import glob
import sys
import copy
import time as t
import psutil


ROWS = 6
COLS = 7

PLAYER_O = 'O'
PLAYER_X = 'X'
EMPTY_SPACE = '-'

debug = False
round = 0
winner = None
show_tree = False
counter = 0

# Evaluation Functions
# Cameron
def eval_function_1(board):
    board_array = board.board  # Convert the board to a 2D array

    def count_groups(board_array, player):
        count = 0
        
        # Check horizontally
        for row in range(ROWS):
            for col in range(COLS - 3):  # Need at least 4 consecutive spaces for a win
                group = [board_array[row][col + i] for i in range(4)]
                count += group.count(player)

        # Check vertically
        for col in range(COLS):
            for row in range(ROWS - 3):  # Need at least 4 consecutive spaces for a win
                group = [board_array[row + i][col] for i in range(4)]
                count += group.count(player)

        # Check diagonally (bottom-left to top-right)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                group = [board_array[row - i][col + i] for i in range(4)]
                count += group.count(player)

        # Check diagonally (top-left to bottom-right)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                group = [board_array[row + i][col + i] for i in range(4)]
                count += group.count(player)

        return count

    max_player_groups = count_groups(board_array, PLAYER_X)
    min_player_groups = count_groups(board_array, PLAYER_O)
    evaluation = max_player_groups - min_player_groups

    return evaluation


# Dhruve
def eval_function_2(board):
    """
    Computes the difference between the number of the three-in-a-row
    pieces for PLAYER_X and PLAYER_O. It checks both horizontally and vertically.

    Args:
    - board (object): The current game board.
    
    Returns:
    - int: The difference in the number of three-in-a-row sequences between PLAYER_X and PLAYER_O.
    """
    def count_threes(player):
        """
        Helper function to count the number of three-in-a-row sequences for a given player.

        Args:
        - player (str): Either PLAYER_X or PLAYER_O.

        Returns:
        - int: The number of three-in-a-row sequences for the given player.
        """
        count = 0
        for row in range(ROWS):
            for col in range(COLS - 2):
                if board.board[row][col:col+3] == [player]*3:
                    count += 1
        for col in range(COLS):
            for row in range(ROWS - 2):
                if board.board[row][col] == board.board[row+1][col] == board.board[row+2][col] == player:
                    count += 1
        return count

    return count_threes(PLAYER_X) - count_threes(PLAYER_O)

# Youssef
def eval_function_3(board):
    board_array = board.board  # Convert the board to a 2D array

    def count_groups(board_array, player):
        count = 0
        # Check horizontally
        for row in range(ROWS):
            for col in range(COLS - 3):  # Need at least 4 consecutive spaces for a win
                group = [board_array[row][col + i] for i in range(4)]
                count += group.count(player)

        # Check vertically
        for col in range(COLS):
            for row in range(ROWS - 3):  # Need at least 4 consecutive spaces for a win
                group = [board_array[row + i][col] for i in range(4)]
                count += group.count(player)

        # Check diagonally (bottom-left to top-right)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                group = [board_array[row - i][col + i] for i in range(4)]
                count += group.count(player)

        # Check diagonally (top-left to bottom-right)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                group = [board_array[row + i][col + i] for i in range(4)]
                count += group.count(player)

        return count

    # Calculate the total number of winning lines for MAX (Player X)
    max_player_groups = count_groups(board_array, PLAYER_X)

    # Calculate the total number of winning lines for the opponent (MIN, Player O)
    min_player_groups = count_groups(board_array, PLAYER_O)

    # Calculate the evaluation using the formula E(n) = M(n) - O(n)
    evaluation = max_player_groups - min_player_groups
    return evaluation


class Metrics:
    def __init__(self):
        self.nodes_generated = 0
        self.nodes_expanded = 0
        self.player_x_time = 0
        self.player_o_time = 0

    def start_timer(self, player):
        if player == PLAYER_X:
            self.player_x_start_time = t.time()
        else:
            self.player_o_start_time = t.time()


    def stop_timer(self, player):
        if player == PLAYER_X:
            self.player_x_time += t.time() - self.player_x_start_time
        else:
            self.player_o_time += t.time() - self.player_o_start_time

    # def elasped_time(self, player):
    #     if player == PLAYER_X:
    #         return self.player_x_time
    #     else:
    #         return self.player_o_time

    
    def increment_nodes_generated(self):
        self.nodes_generated += 1
    
    def increment_nodes_expanded(self):
        self.nodes_expanded += 1

    
class Board:
    def __init__(self):
        self.rows = ROWS
        self.columns = COLS
        self.board = [[EMPTY_SPACE for _ in range(COLS)] for _ in range(ROWS)]
        self.turn = PLAYER_X

    def winner(self):
        """ Return the winner of the game. """
        bool = self.check_win()
        if self.turn == PLAYER_O:
            return PLAYER_X if bool else None
        if self.turn == PLAYER_X:
            return PLAYER_O if bool else None
        
        return None
    

    def display(self, file=None):
        output_stream = file if file is not None else sys.stdout
        global round, winner
        print("\n" + f"Round: {round + 1}", file=output_stream)
        print(f"Player {PLAYER_X if round % 2 == 0 else PLAYER_O} turn", file=output_stream)
        print("-" * (COLS * 4 + 1), file=output_stream)

        winner = self.winner()
        
        for row in self.board:
            print("|", end=" ", file=output_stream)
            for cell in row:
                print(cell, end=" | ", file=output_stream)
            print("\n" + "-" * (COLS * 4 + 1), file=output_stream)
        if winner == PLAYER_O or winner == PLAYER_X:
            print(f"We have a winner! Player {winner} won!", file=output_stream)
        else:
            print("No player wins. It was a tie.", file=output_stream)

    def place_token(self, column):
        for i in range(self.rows - 1, -1, -1):
            if self.board[i][column] == EMPTY_SPACE:
                self.board[i][column] = self.turn
                self.switch_turn()
                return True
        return False

    def switch_turn(self):
        self.turn = PLAYER_O if self.turn == PLAYER_X else PLAYER_X

    def is_terminal(self):
        return self.check_win() or all(cell != EMPTY_SPACE for row in self.board for cell in row)

    def check_win(self):
        """
        Check if there's a winning move on the board.
        A win is defined by four consecutive pieces of the same type a row, column, or diagonal.
        
        Returns:
        - bool: True if a winning move exists, otherwise False.
        """
        # Check the rows for 4 consecutive pieces of the same type
        for row in range(ROWS):
            for col in range(COLS - 3): # Subtract 3 to prevent index out of the bound error
                if self.board[row][col] == self.board[row][col + 1] == self.board[row][col + 2] == self.board[row][col + 3] and self.board[row][col] != EMPTY_SPACE:
                    return True
                
        # Check the columns for 4 consecutive pieces of the same type
        for col in range(COLS):
            for row in range(ROWS - 3): # Subtract 3 to prevent index out of the bound error
                if self.board[row][col] == self.board[row + 1][col] == self.board[row + 2][col] == self.board[row + 3][col] and self.board[row][col] != EMPTY_SPACE:
                    return True
                
        # Check the diagonals for 4 consecutive pieces of the same type
        for row in range(ROWS - 3): # Subtract 3 to prevent index out of the bound error
            for col in range(COLS - 3): # Subtract 3 to prevent index out of the bound error
                # Check top-left to bottom-right diagonal
                if self.board[row][col] == self.board[row + 1][col + 1] == self.board[row + 2][col + 2] == self.board[row + 3][col + 3] and self.board[row][col] != EMPTY_SPACE:
                    return True
                # Check bottom-left to top-right diagonal
                if self.board[row + 3][col] == self.board[row + 2][col + 1] == self.board[row + 1][col + 2] == self.board[row][col + 3] and self.board[row + 3][col] != EMPTY_SPACE:
                    return True
                
        # No winning move found
        return False


def format_board_for_logging(board):
    """ Format the board state into a string representation for logging. """
    board_str = ""
    for row in board:
        board_str += ' '.join(row) + '\n'
    return board_str

def show_algorithm_tree(board, current_depth, scenario_counter):
    global round, counter
    if show_tree:
        with open(os.path.join("Project2_AI", "SearchTree", f"scenario_{scenario_counter + 1}",f"round_{round + 1}.txt"), "a") as file:
            if current_depth == 1:
                file.write(f"\n=== New Board. Player {board.turn} ===\n")
            file.write(f"Depth: {current_depth}\n")
            file.write(format_board_for_logging(board.board))
            file.write("\n" + "-"*20 + "\n")  # Separator
  
    
def minimax_alpha_beta(board, current_depth, max_depth, current_player, maximizing_player, alpha, beta, eval_function, metrics, scenario):
    """
    Implement the Minimax algorithm with alpha-beta pruning to find the best move for a given player.

    Parameters:
    - board: The current game state.
    - depth: The current depth in the search tree.
    - current_player: The player whose turn it currently is.
    - maximizing_player: A flag indicating if the current move is a maximizing move.
    - alpha: The best value achieved so far by any choice the maximizer has made at any choice point along the path.
    - beta: The smallest value achieved so far by any choice the minimizer has made at any choice point along the path.
    - eval_function: The evaluation function used to evaluate the board state.

    Returns:
    - The best move's value from the current state.

    """
    global round, counter, show_tree
    counter += 1
    show_algorithm_tree(board, current_depth, scenario)
    # print(f"{current_depth}   ", end='')
    metrics.increment_nodes_expanded()
    # Best case: If we've reached the maximum depth or the board state is terminal
    if current_depth >= max_depth or board.is_terminal():
        # print(f"Board returned at Round: {round}")
        # Return the evaluation for the current board. Negate for 0 since it's the minimizing player
        return eval_function(board) if current_player == PLAYER_X else -eval_function(board)

    # Maximizing player's logic
    if maximizing_player:
        max_eval = float('-inf') # Initialiaze to negative infinity
        for col in range(board.columns):
            # Create a temporary copy of the board to simulate the move
            temp_board = copy.deepcopy(board)
            temp_board.place_token(col)

            # Recursive call to continute the search tree
            metrics.increment_nodes_generated()
            eval = minimax_alpha_beta(temp_board, current_depth + 1, max_depth, board.turn, False, alpha, beta, eval_function, metrics, scenario)
            max_eval = max(max_eval, eval)
            # Alpha-beta pruning logic
            alpha = max(alpha, eval)
            if beta <= alpha: # If beta is less than or equal to alpha, prune the branch
                break
        return max_eval
    
    # Minimizing player's logic
    else:
        min_eval = float('inf') # Initialiaze to positive infinity
        for col in range(board.columns):
            # Create a temporary copy of the board to simulate the move
            temp_board = copy.deepcopy(board)
            temp_board.place_token(col)

            # Recursive call to continue the search tree
            metrics.increment_nodes_generated()
            eval = minimax_alpha_beta(temp_board, current_depth + 1, max_depth, board.turn, True, alpha, beta, eval_function, metrics, scenario)
            min_eval = min(min_eval, eval)

            # Alpha-beta pruning logic
            beta = min(beta, eval)
            if beta <= alpha: # If beta is less than or equal to alpha, prune the branch
                break
        return min_eval

def get_best_move(board, max_depth, eval_function, metrics, scenario):
    best_move = -1
    best_value = float('-inf') if board.turn == PLAYER_X else float('inf')
    valid_moves = [col for col in range(board.columns) if board.board[0][col] == EMPTY_SPACE]  # Only consider columns that aren't full

    for col in valid_moves:
        temp_board = copy.deepcopy(board)
        temp_board.place_token(col)

        move_value = minimax_alpha_beta(temp_board, 1, max_depth, board.turn, board.turn == PLAYER_O, float('-inf'), float('inf'), eval_function, metrics, scenario)
            
        if board.turn == PLAYER_X and move_value > best_value:
            best_value = move_value
            best_move = col
        elif board.turn == PLAYER_O and move_value < best_value:
            best_value = move_value
            best_move = col

    return best_move


def print_metrics_to_file(metrics1, metrics2, filename, eval1, eval2, total_time, mem_used, scenario_counter, board):
    """
    Write a comparison of metrics for two avaluation functions to its respective file.
    Uses a table style format.
    
    Parameters:
    - metrics1: The metrics gathered from the first eval function.
    - metrics2: The metrics gathered from the second eval function.
    - filename: The name of the file where the comparison table will be appended on.
    """

    global counter
    header = "{:<25} {:<20} {:<20} {:<20} {:<20}\n".format("Metric Used", eval1, eval2, "Difference", "Who did better?")
    divider = "-" * 105 + "\n"

    nodes_gen_better = eval1 if metrics1.nodes_generated < metrics2.nodes_generated else eval2
    nodes_exp_better = eval1 if metrics1.nodes_expanded < metrics2.nodes_expanded else eval2
    # print(f"Nodes Better: {nodes_exp_better} Super Meh: {nodes_gen_better}")
    if metrics1.player_x_time > metrics2.player_o_time:
        time_better = eval2
    elif metrics1.player_x_time < metrics2.player_o_time:
        time_better = eval1
    else:
        time_better = "No one."
    
    with open(filename, "a") as file:
        file.write(header)
        file.write(divider)
        file.write("{:<25} {:<20} {:<20} {:<20} {:<20}\n".format("Nodes Generated", metrics1.nodes_generated, metrics2.nodes_generated, metrics1.nodes_generated - metrics2.nodes_generated, nodes_gen_better))
        file.write("{:<25} {:<20} {:<20} {:<20} {:<20}\n".format("Nodes Expanded", metrics1.nodes_expanded, metrics2.nodes_expanded, metrics1.nodes_expanded - metrics2.nodes_expanded, nodes_exp_better))
        file.write("{:<25} {:<20.4f} {:<20.4f} {:<20.4f} {:<20}\n".format("Elapsed Time", metrics1.player_x_time, metrics2.player_o_time, metrics1.player_x_time - metrics2.player_o_time, time_better))
    
    # Write the tabulated results at the end
    with open(os.path.join("Project2_AI", "tabulation", "analysis.txt"), "a") as file:
        file.write(f"Results for Scenario: {scenario_counter + 1}\n")
        file.write(f"Winner for this case: {board.winner()}\n")
        file.write(header)
        file.write(divider)
        file.write("{:<25} {:<20} {:<20} {:<20} {:<20}\n".format("Nodes Generated", metrics1.nodes_generated, metrics2.nodes_generated, metrics1.nodes_generated - metrics2.nodes_generated, nodes_gen_better))
        file.write("{:<25} {:<20} {:<20} {:<20} {:<20}\n".format("Nodes Expanded", metrics1.nodes_expanded, metrics2.nodes_expanded, metrics1.nodes_expanded - metrics2.nodes_expanded, nodes_exp_better))
        file.write("{:<25} {:<20.4f} {:<20.4f} {:<20.4f} {:<20}\n".format("Elapsed Time", metrics1.player_x_time, metrics2.player_o_time, metrics1.player_x_time - metrics2.player_o_time, time_better))
        file.write(f"Total Time Elasped {total_time} seconds.\n")
        file.write(f"Total Memory Used {mem_used} MB.\n")
        file.write(f"Total Boards Evaluated: {counter}\n")
        file.write('=' * 105 + '\n\n')
        
# Function removes all the contents in the directory
def remove_txt_files(directory):
    txt_files = os.path.join(directory, '**', '*.txt')
    txt_files = glob.glob(txt_files, recursive=True)
    for txt_file in txt_files:
        os.remove(txt_file)

def main():
    # Remove, only keep when using visual studio
    sys.argv.insert(1,'--nogui')
    sys.argv.insert(2,'--notree')
    
    # Define scenarios
    scenarios = [
        {"max_func": eval_function_1, "max_depth": 2, "min_func": eval_function_2, "min_depth": 2}, # Eval 1 v 2
        {"max_func": eval_function_1, "max_depth": 2, "min_func": eval_function_3, "min_depth": 4}, # Eval 1 v 3
        {"max_func": eval_function_2, "max_depth": 2, "min_func": eval_function_3, "min_depth": 8}, # Eval 2 v 3

        {"max_func": eval_function_1, "max_depth": 4, "min_func": eval_function_2, "min_depth": 2}, # Eval 1 v 2
        {"max_func": eval_function_1, "max_depth": 4, "min_func": eval_function_3, "min_depth": 4}, # Eval 1 v 3
        {"max_func": eval_function_2, "max_depth": 4, "min_func": eval_function_3, "min_depth": 8}, # Eval 2 v 3

        {"max_func": eval_function_1, "max_depth": 8, "min_func": eval_function_2, "min_depth": 2}, # Eval 1 v 2
        {"max_func": eval_function_1, "max_depth": 8, "min_func": eval_function_3, "min_depth": 4}, # Eval 1 v 3
        {"max_func": eval_function_2, "max_depth": 8, "min_func": eval_function_3, "min_depth": 8}, # Eval 2 v 3

    ]
    output_dir = os.path.join("Project2_AI", "Outputs")
    remove_txt_files(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_tree_dir = os.path.join("Project2_AI", "SearchTree")
    remove_txt_files(output_tree_dir)
    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
        
    tabulation_dir = os.path.join("Project2_AI", "tabulation")
    remove_txt_files(tabulation_dir)
    if not os.path.exists(tabulation_dir):
        os.makedirs(tabulation_dir)
    
    global round, counter, show_tree
    if '--notree' in sys.argv[2]:
        print("No tree will be generated")
        show_tree = False
    else:
        print("Tree will be generated. Please check the \"SearchTree\" folder")
        show_tree = True
        
        
    for i, scenario in enumerate(scenarios):
        counter = 0
        start_time = t.time()
        metrics_max = Metrics()
        metrics_min = Metrics()
        board = Board()
        scenarios_tree = os.path.join(output_tree_dir, f"scenario_{i + 1}")
        if not os.path.exists(scenarios_tree):
            os.makedirs(scenarios_tree)
            
        # Continue the loop until winner or board is full
        while not board.is_terminal():
            if board.turn == PLAYER_X:
                metrics_max.start_timer(PLAYER_X)
                move = get_best_move(board, scenario["max_depth"], scenario["max_func"], metrics_max, i)
                metrics_max.stop_timer(PLAYER_X)
            else:
                metrics_min.start_timer(PLAYER_O)
                move = get_best_move(board, scenario["min_depth"], scenario["min_func"], metrics_min, i)
                metrics_min.stop_timer(PLAYER_O)
            move_made = board.place_token(move)
            if not move_made:
                print("Invalid move attempted")
                break
            output_file_name = os.path.join(output_dir, f"output_scenario{i + 1}.txt")
            with open(output_file_name, "a") as output_file:
                board.display(output_file)
            round += 1

        memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        print(f"Evaluated {counter} boards.")
        end_time = t.time()
        start_to_end = '%.2f' % (end_time - start_time)
        print(f"Scenario {i + 1} took {start_to_end} seconds.")
        print(f"Memory Used: {memory_used} MB")
        
        print_metrics_to_file(metrics_max, metrics_min, output_file_name, scenario["max_func"].__name__, scenario["min_func"].__name__, start_to_end, memory_used, i, board)
        counter = 0
        round = 0
    
    # Check if user would like GUI at the end
    if '--nogui' in sys.argv[1]:
        print("No GUI, please check analysis.txt to view the results instead.")
    else:
        print("Recongized user wants GUI. Although no function therefore just check analysis.txt")
        

if __name__ == "__main__":
    main()
