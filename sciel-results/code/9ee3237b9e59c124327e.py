import random
import time

def construct():
    n = 10
    global_limit = time.time() + 55.0

    # Fixed base square: cyclic Latin square
    # S1[i][j] = (i + j) % n
    S1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    # We will try to build a set of MOLS.
    # Start with S1.
    # We use a randomized backtracking search for each new square.
    # To maximize the count, we restart the entire process if we get stuck,
    # but we keep the best result found so far.
    
    best_squares = []
    
    # Strategy:
    # 1. Try to extend the set starting from S1.
    # 2. If we get stuck at k squares, restart with a different randomization seed/order.
    # 3. If we find a set larger than current best, update best.
    # 4. Since we know 9 MOLS exist, we aim for 9.
    
    # To speed up, we can fix the first few rows of the new square?
    # No, that might constrain too much.
    # Instead, we use a heuristic ordering for cells and values.
    
    def find_orthogonal(squares, timeout=5.0):
        """
        Finds a Latin square orthogonal to all squares in 'squares'.
        Uses randomized backtracking with MRV (Minimum Remaining Values) heuristic.
        """
        grid = [[-1] * n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        
        # pair_used[k] is a set of (val_in_square_k, val_in_new_square)
        pair_used = [set() for _ in squares]
        
        start_time = time.time()
        
        # Pre-compute the "cost" or just use random ordering?
        # MRV is better.
        
        # We need to track which cells are empty.
        # Since n is small (10), we can just scan for the best cell each time.
        
        def backtrack():
            if time.time() - start_time > timeout:
                return False
            
            # Find the empty cell with the fewest valid options
            min_options = n + 1
            best_i, best_j = -1, -1
            best_candidates = []
            
            for i in range(n):
                for j in range(n):
                    if grid[i][j] != -1:
                        continue
                    
                    # Calculate candidates
                    candidates = []
                    for v in range(n):
                        if v in row_used[i] or v in col_used[j]:
                            continue
                        # Check orthogonality
                        ok = True
                        for k, sq in enumerate(squares):
                            if (sq[i][j], v) in pair_used[k]:
                                ok = False
                                break
                        if ok:
                            candidates.append(v)
                    
                    if len(candidates) < min_options:
                        min_options = len(candidates)
                        best_i, best_j = i, j
                        best_candidates = candidates
                        if min_options == 0:
                            return False
                        if min_options == 1:
                            break
                
                if min_options == 1:
                    break
            
            if best_i == -1:
                # No empty cells? Should not happen unless grid is full
                # If grid is full, we succeeded
                return True
            
            # Shuffle candidates to randomize
            random.shuffle(best_candidates)
            
            for v in best_candidates:
                # Assign
                grid[best_i][best_j] = v
                row_used[best_i].add(v)
                col_used[best_j].add(v)
                for k, sq in enumerate(squares):
                    pair_used[k].add((sq[best_i][best_j], v))
                
                if backtrack():
                    return True
                
                # Unassign
                grid[best_i][best_j] = -1
                row_used[best_i].remove(v)
                col_used[best_j].remove(v)
                for k, sq in enumerate(squares):
                    pair_used[k].remove((sq[best_i][best_j], v))
            
            return False

        if backtrack():
            return [row[:] for row in grid]
        return None

    # Main loop
    # We try multiple times with different random seeds to escape local minima
    # or difficult instances.
    
    # First, try to build from S1.
    current_squares = [S1]
    
    # We want to add up to 9 squares.
    # Let's try to add them one by one.
    # If we fail to add the next one, we might restart the whole search?
    # Or just try again with a different timeout?
    # Given the time limit, restarting the whole thing is safer if we are stuck early.
    
    # Let's structure it as:
    # For attempt in range(max_attempts):
    #   squares = [S1]
    #   While len(squares) < 9 and time_left > min_time:
    #      next_sq = find_orthogonal(squares)
    #      if next_sq: squares.append(next_sq)
    #      else: break
    #   if len(squares) > len(best_squares): best_squares = squares
    #   if len(best_squares) == 9: break

    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > global_limit - 5.0:
            break
            
        squares = [S1]
        
        # Try to extend
        while len(squares) < 9:
            remaining = global_limit - time.time()
            if remaining < 2.0:
                break
            
            # Allocate a portion of the remaining time to this step
            # If we are adding the 2nd square, we can be generous.
            # If we are adding the 9th, we need to be careful.
            # Let's use a fixed small timeout for each step to allow many restarts.
            timeout = 2.0
            if len(squares) == 1:
                timeout = 5.0
            
            next_sq = find_orthogonal(squares, timeout=timeout)
            
            if next_sq is None:
                break
            
            squares.append(next_sq)
            
        if len(squares) > len(best_squares):
            best_squares = squares
            if len(best_squares) == 9:
                break
                
    # If we didn't find anything better than 1 (which shouldn't happen as S1 is valid),
    # return S1.
    if not best_squares:
        best_squares = [S1]
        
    # Ensure types are correct
    # They are lists of lists of ints.
    
    return best_squares
