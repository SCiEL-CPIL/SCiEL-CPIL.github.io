import time
import random

def construct():
    n = 10
    deadline = time.time() + 50  # 50 second safety margin
    
    # First Latin square: cyclic
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We need to find L2 such that:
    # 1. L2 is a Latin square
    # 2. For each diagonal d (i+j ≡ d mod n), L2 takes each value 0..n-1 exactly once
    #    (this ensures orthogonality with L1)
    
    # Represent L2 as a 2D array, initially empty
    L2 = [[-1] * n for _ in range(n)]
    
    # For constraint propagation, track:
    # - row_used[i][v] = True if value v is used in row i
    # - col_used[j][v] = True if value v is used in column j
    # - diag_used[d][v] = True if value v is used on diagonal d (where d = (i+j)%n)
    
    row_used = [[False] * n for _ in range(n)]
    col_used = [[False] * n for _ in range(n)]
    diag_used = [[False] * n for _ in range(n)]
    
    # Count of empty cells
    empty_count = n * n
    
    def is_valid(i, j, v):
        d = (i + j) % n
        return not row_used[i][v] and not col_used[j][v] and not diag_used[d][v]
    
    def place(i, j, v):
        d = (i + j) % n
        L2[i][j] = v
        row_used[i][v] = True
        col_used[j][v] = True
        diag_used[d][v] = True
    
    def remove(i, j, v):
        d = (i + j) % n
        L2[i][j] = -1
        row_used[i][v] = False
        col_used[j][v] = False
        diag_used[d][v] = False
    
    # Order cells by most constrained first (fewest valid values)
    # We'll use a dynamic ordering
    
    def get_valid_values(i, j):
        d = (i + j) % n
        vals = []
        for v in range(n):
            if not row_used[i][v] and not col_used[j][v] and not diag_used[d][v]:
                vals.append(v)
        return vals
    
    # Backtracking search
    def backtrack():
        nonlocal empty_count
        if time.time() > deadline:
            return False
        
        if empty_count == 0:
            return True
        
        # Find the empty cell with fewest valid values (MRV heuristic)
        min_count = n + 1
        best_i, best_j = -1, -1
        best_vals = []
        
        for i in range(n):
            for j in range(n):
                if L2[i][j] == -1:
                    vals = get_valid_values(i, j)
                    if len(vals) < min_count:
                        min_count = len(vals)
                        best_i, best_j = i, j
                        best_vals = vals
                        if min_count == 0:
                            return False
        
        if best_i == -1:
            return True  # No empty cells
        
        # Try values in random order for diversity
        random.shuffle(best_vals)
        for v in best_vals:
            place(best_i, best_j, v)
            empty_count -= 1
            if backtrack():
                return True
            remove(best_i, best_j, v)
            empty_count += 1
        
        return False
    
    # Try multiple random restarts if needed
    # First, try with the natural ordering
    if backtrack():
        # Convert to ensure Python ints
        L2_final = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
        L1_final = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
        return [L1_final, L2_final]
    
    # If backtracking fails (shouldn't happen if we have enough time), try randomized approach
    # Randomized local search
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Initialize L2 randomly
        L2_rand = [[-1] * n for _ in range(n)]
        row_used = [[False] * n for _ in range(n)]
        col_used = [[False] * n for _ in range(n)]
        diag_used = [[False] * n for _ in range(n)]
        
        # Fill randomly with constraint checking
        cells = [(i, j) for i in range(n) for j in range(n)]
        random.shuffle(cells)
        
        success = True
        for (i, j) in cells:
            d = (i + j) % n
            valid = [v for v in range(n) if not row_used[i][v] and not col_used[j][v] and not diag_used[d][v]]
            if not valid:
                success = False
                break
            v = random.choice(valid)
            L2_rand[i][j] = v
            row_used[i][v] = True
            col_used[j][v] = True
            diag_used[d][v] = True
        
        if success:
            L2_final = [[int(L2_rand[i][j]) for j in range(n)] for i in range(n)]
            L1_final = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
            return [L1_final, L2_final]
    
    # Fallback: return just the base square
    base_square = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [base_square]
