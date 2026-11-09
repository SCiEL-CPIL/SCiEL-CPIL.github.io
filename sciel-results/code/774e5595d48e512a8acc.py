import time
import random

def construct():
    # Deadline guard: 90s wall-clock limit, use 85s as safety margin
    deadline = time.time() + 85.0
    n = 10
    
    # Construct a standard Latin square of order 10
    # L[i][j] = (i + j) % n
    # This is a valid Latin square: each row and column contains 0..9 exactly once.
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find a second square L2 orthogonal to L1 using backtracking.
    # State: current row i, column j
    #        row_used[i] (set of symbols used in row i)
    #        col_used[j] (set of symbols used in col j)
    #        pairs_used (set of (L1[i][j], L2[i][j]) tuples)
    
    # We will fill L2 row by row.
    # To improve performance, we can fix the first row of L2 to be the identity permutation
    # (0, 1, ..., 9). This is without loss of generality for finding *an* orthogonal square
    # because we can permute the symbols of L2.
    # However, let's just do a standard backtracking search with some heuristics.
    
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    row_symbols = [set() for _ in range(n)]
    col_symbols = [set() for _ in range(n)]
    used_pairs = set()
    
    # Precompute L1 values for quick access
    # l1_vals[i][j] is already L1[i][j]
    
    # Heuristic: Order cells by most constrained first?
    # For Latin squares, row/col constraints are symmetric.
    # Let's try filling row by row, column by column.
    
    # Optimization: Precompute available symbols for each row/col?
    # Since n is small (10), sets are fast enough.
    
    # Let's add a time check inside the recursion to avoid deep recursion timeouts
    # if it gets stuck.
    
    def backtrack(i, j):
        if time.time() > deadline - 5.0:
            return False
        
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
        
        # Try all symbols s in 0..n-1
        # We can optimize the order of symbols, but let's keep it simple first.
        # Maybe shuffle the order to avoid deterministic worst-cases?
        # No, deterministic is fine for reproducibility unless we want random restarts.
        
        # A better heuristic: choose the next cell with the fewest candidates.
        # But implementing MRV (Minimum Remaining Values) dynamically is complex.
        # Let's stick to row-major order but check if we can prune early.
        
        for s in range(n):
            if s in row_symbols[i] or s in col_symbols[j]:
                continue
            pair = (L1[i][j], s)
            if pair in used_pairs:
                continue
            
            # Place s
            L2[i][j] = s
            row_symbols[i].add(s)
            col_symbols[j].add(s)
            used_pairs.add(pair)
            
            if backtrack(i, j + 1):
                return True
            
            # Undo
            L2[i][j] = -1
            row_symbols[i].remove(s)
            col_symbols[j].remove(s)
            used_pairs.remove(pair)
            
        return False
    
    # Try the backtracking search
    found = backtrack(0, 0)
    
    if found:
        # Ensure types are Python ints and lists
        # L1 and L2 are already lists of lists of ints
        result = [L1, L2]
        return result
    else:
        # Fallback: return just L1
        return [L1]
