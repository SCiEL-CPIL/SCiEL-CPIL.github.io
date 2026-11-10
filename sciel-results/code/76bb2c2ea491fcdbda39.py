import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = 10
    # S0 is always valid
    s0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [s0]
    
    # State for backtracking
    # row_used[i][val]
    # col_used[j][val]
    # pair_used[val1][val2] for each existing square
    
    def find_next_square(existing_squares, time_limit):
        """
        Finds a Latin square orthogonal to all existing squares.
        Returns the square or None.
        """
        n = 10
        # Pre-compute pair constraints for each existing square
        # pair_constraints[k][val1][val2] = True if (val1, val2) is used in square k
        # Actually, we just need to ensure the NEW square's pairs with EACH existing square are unique.
        
        # For each existing square S_e, we have a used_pairs[e] set.
        # When we place a value v in the new square at (i,j), we must check:
        # 1. v not in row i
        # 2. v not in col j
        # 3. For each existing square S_e, the pair (S_e[i][j], v) is not used in new_pairs_e.
        
        # We can use backtracking.
        # To make it fast, we iterate cells in a fixed order (row by row) or use MRV.
        # For n=10, fixed order might be okay if we prune well.
        
        new_square = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [[False] * n for _ in range(n)]
        col_used = [[False] * n for _ in range(n)]
        
        # For each existing square, track used pairs
        # existing_squares is a list of lists of lists
        # new_pairs[e] is a 2D list of sets or bitmasks?
        # n=10, so 100 pairs. A set of tuples is fine.
        new_pairs = [set() for _ in existing_squares]
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            
            if idx == n * n:
                return True
            
            i = idx // n
            j = idx % n
            
            # Try all values 0..9
            # Optimization: Check constraints first
            for v in range(n):
                if row_used[i][v] or col_used[j][v]:
                    continue
                
                # Check orthogonality constraints
                valid = True
                for e_idx, s_e in enumerate(existing_squares):
                    val_e = s_e[i][j]
                    pair = (val_e, v)
                    if pair in new_pairs[e_idx]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Place v
                new_square[i][j] = v
                row_used[i][v] = True
                col_used[j][v] = True
                for e_idx in range(len(existing_squares)):
                    val_e = existing_squares[e_idx][i][j]
                    new_pairs[e_idx].add((val_e, v))
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                new_square[i][j] = -1
                row_used[i][v] = False
                col_used[j][v] = False
                for e_idx in range(len(existing_squares)):
                    val_e = existing_squares[e_idx][i][j]
                    new_pairs[e_idx].remove((val_e, v))
            
            return False
        
        # Use a random shuffle of values to avoid deterministic failures?
        # Or just try 0..9.
        
        if backtrack(0):
            return new_square
        return None

    # Try to find S1
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        s1 = find_next_square(squares, deadline - 5)
        if s1:
            squares.append(s1)
    
    # Try to find S2
    remaining_time = deadline - time.time()
    if remaining_time > 5 and len(squares) == 2:
        s2 = find_next_square(squares, deadline - 5)
        if s2:
            squares.append(s2)
            
    # Try to find S3
    remaining_time = deadline - time.time()
    if remaining_time > 5 and len(squares) == 3:
        s3 = find_next_square(squares, deadline - 5)
        if s3:
            squares.append(s3)

    # Convert to Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
