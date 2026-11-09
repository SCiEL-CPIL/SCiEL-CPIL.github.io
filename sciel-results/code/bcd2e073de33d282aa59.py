import time

def construct():
    # Deadline guard: 90s wall-clock limit, use 85s as safety margin
    deadline = time.time() + 85.0
    
    n = 10
    # Construct a standard Latin square of order 10
    # L[i][j] = (i + j) % n
    # This is a valid Latin square: each row and column contains 0..9 exactly once.
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We want to find L2 orthogonal to L1.
    # L2[i][j] must be a permutation of 0..n-1 in each row and column.
    # And the pair (L1[i][j], L2[i][j]) must be unique for all i,j.
    
    # Use backtracking with time limit.
    start_time = time.time()
    time_limit = 80  # seconds
    
    # Initialize L2
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Precompute L1 values
    l1_vals = L1
    
    # We will fill L2 row by row.
    # row_symbols[i] = set of symbols used in row i
    # col_symbols[j] = set of symbols used in col j
    # used_pairs = set of (l1, l2) pairs
    
    row_symbols = [set() for _ in range(n)]
    col_symbols = [set() for _ in range(n)]
    used_pairs = set()
    
    def backtrack(i, j):
        if time.time() - start_time > time_limit:
            return False
        if i == n:
            return True
        if j == n:
            return backtrack(i + 1, 0)
        
        # Try all symbols s in 0..n-1
        for s in range(n):
            if s in row_symbols[i] or s in col_symbols[j]:
                continue
            pair = (l1_vals[i][j], s)
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
    
    if backtrack(0, 0):
        # Convert to list of lists of ints (ensure Python int)
        L2 = [[int(L2[i][j]) for j in range(n)] for i in range(n)]
        L1 = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
        return [L1, L2]
    else:
        # Fallback: return just L1
        L1 = [[int(L1[i][j]) for j in range(n)] for i in range(n)]
        return [L1]
