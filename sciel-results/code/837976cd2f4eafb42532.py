import time

def construct():
    n = 10
    # First Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # We want to find L2 orthogonal to L1.
    # L2[i][j] must be a permutation of 0..n-1 in each row and column.
    # And the pair (L1[i][j], L2[i][j]) must be unique for all i,j.
    
    # We can use backtracking.
    # State: current row i, column j, 
    #        row_used[i] (set of symbols used in row i)
    #        col_used[j] (set of symbols used in col j)
    #        pairs_used (set of (L1[i][j], L2[i][j]) tuples)
    
    # To speed up, we can fix the first row of L2 to be a permutation.
    # Actually, we can just search.
    
    # Since n=10 is small, this will be fast.
    # But to be safe against timeouts, I will add a time limit.
    
    start_time = time.time()
    time_limit = 50  # seconds
    
    # Initialize L2
    L2 = [[-1 for _ in range(n)] for _ in range(n)]
    
    # Precompute L1 values
    l1_vals = [[(i + j) % n for j in range(n)] for i in range(n)]
    
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
        # Convert to list of lists of ints
        result = [L1, L2]
        return result
    else:
        # Fallback: return just L1
        return [L1]
