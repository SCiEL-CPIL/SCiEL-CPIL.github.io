import time
import sys

def construct():
    n = 10
    # L1: Cyclic Latin Square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1]
    
    start_time = time.time()
    
    # Try to find L2
    # We will use a backtracking solver to find a square orthogonal to L1
    L2 = find_next_square(squares, n, timeout=10)
    if L2:
        squares.append(L2)
        
    # Try to find L3
    # This is likely to fail or take longer if N(10)=2
    if time.time() - start_time < 50:
        L3 = find_next_square(squares, n, timeout=40)
        if L3:
            squares.append(L3)
            
    return squares

def find_next_square(fixed_squares, n, timeout=10):
    start_time = time.time()
    
    # We will use a backtracking algorithm with forward checking.
    # State:
    # grid: n x n list, initialized to -1
    # row_used: n x n boolean matrix (row i, symbol v)
    # col_used: n x n boolean matrix (col j, symbol v)
    # ortho_counts: list of dicts or matrices for each fixed square
    # For each fixed square L_fix, we need to ensure (L_fix[i][j], grid[i][j]) are unique.
    # So for each L_fix, we maintain a count of how many times value v has been placed
    # in cells where L_fix has symbol s.
    # Actually, it's easier to maintain a set of seen pairs for each fixed square?
    # No, we need to prune.
    # Let's maintain for each fixed square L_fix:
    # ortho_used[L_fix][s][v] = count of v in grid at cells where L_fix is s.
    # Since we need distinct pairs, for a fixed s, each v can appear at most once.
    # So ortho_used[L_fix][s][v] must be 0 or 1.
    
    grid = [[-1] * n for _ in range(n)]
    row_used = [[False] * n for _ in range(n)]
    col_used = [[False] * n for _ in range(n)]
    
    # Precompute L1 symbols for each cell to speed up ortho check
    # L_fix_symbols[k][i][j] is the symbol in fixed_squares[k] at (i,j)
    # And we need to track counts.
    # Let's use a list of 2D arrays for ortho counts.
    ortho_counts = []
    for L_fix in fixed_squares:
        # ortho_counts[k][s][v]
        # s is the symbol in L_fix, v is the value in grid
        # We need an array of size n x n for each k?
        # Actually, for a fixed L_fix, the constraint is:
        # For each s in 0..n-1, the values v in grid[i][j] where L_fix[i][j]==s must be distinct.
        # So we can have a 2D array `counts` of size n x n.
        # counts[s][v] = 1 if v has been used in a cell where L_fix is s.
        counts = [[0] * n for _ in range(n)]
        ortho_counts.append(counts)
        
    # Variable ordering:
    # We can try to fill cells in a specific order.
    # A good heuristic is to pick the cell with the fewest available values (MRV).
    # However, implementing dynamic MRV in Python might be slow if not careful.
    # Let's try a static order first: row by row.
    # If that fails, we might need MRV.
    # Given n=10, row-by-row with forward checking might be sufficient for L2.
    
    # Let's implement a recursive backtracking with a timeout check.
    # To avoid deep recursion limits or overhead, we can use an iterative stack or just recursion
    # since depth is 100.
    
    # To speed up, we can fix the first row of the new square to be 0..n-1?
    # This is valid because we can permute the symbols of the new square.
    # Does this affect orthogonality?
    # If we permute symbols of L_new, the pairs (L_fix, L_new) are permuted in the second component.
    # Distinctness is preserved.
    # So we can fix L_new[0][0]=0, L_new[0][1]=1, ..., L_new[0][n-1]=n-1.
    # This reduces the search space by factor of n! (3.6M for n=10).
    # Let's do this.
    
    for j in range(n):
        grid[0][j] = j
        row_used[0][j] = True
        col_used[j][j] = True
        for k in range(len(fixed_squares)):
            s = fixed_squares[k][0][j]
            ortho_counts[k][s][j] = 1
            if ortho_counts[k][s][j] > 1:
                return None # Should not happen if fixed row is valid, but check just in case
                
    # Now solve for rows 1 to n-1
    # We'll use a helper function to check validity and backtrack.
    
    def solve(row):
        if time.time() - start_time > timeout:
            return False
        if row == n:
            return True
            
        # Get available values for this row
        # We can try to use MRV for columns in this row?
        # Or just iterate columns 0 to n-1.
        # Let's try columns 0 to n-1.
        
        for col in range(n):
            # Find valid values for grid[row][col]
            valid_vals = []
            for v in range(n):
                if not row_used[row][v] and not col_used[col][v]:
                    # Check ortho constraints
                    ok = True
                    for k in range(len(fixed_squares)):
                        s = fixed_squares[k][row][col]
                        if ortho_counts[k][s][v] > 0:
                            ok = False
                            break
                    if ok:
                        valid_vals.append(v)
                        
            if not valid_vals:
                return False
                
            # Try each valid value
            # Heuristic: try values that appear least often in other rows? 
            # Or just random?
            # Let's try in order 0..n-1 for determinism, or shuffle for variety?
            # Deterministic is safer for reproducibility, but might get stuck in bad branches.
            # Let's shuffle valid_vals to break symmetry if any.
            import random
            random.shuffle(valid_vals)
            
            for v in valid_vals:
                # Place v
                grid[row][col] = v
                row_used[row][v] = True
                col_used[col][v] = True
                for k in range(len(fixed_squares)):
                    s = fixed_squares[k][row][col]
                    ortho_counts[k][s][v] += 1
                    
                if solve(row + 1 if col == n - 1 else row):
                    return True
                # Wait, my solve function takes 'row'. 
                # If I'm in the middle of a row, I need to go to next col.
                # This recursive structure is a bit messy with just 'row'.
                # Let's change solve to take (row, col).
                
                # Backtrack
                grid[row][col] = -1
                row_used[row][v] = False
                col_used[col][v] = False
                for k in range(len(fixed_squares)):
                    s = fixed_squares[k][row][col]
                    ortho_counts[k][s][v] -= 1
                    
        return False
        
    # The above solve function is flawed because it doesn't handle column iteration correctly in recursion.
    # Let's rewrite solve to take (row, col).
    
    def solve_rc(row, col):
        if time.time() - start_time > timeout:
            return False
            
        if row == n:
            return True
        if col == n:
            return solve_rc(row + 1, 0)
            
        # If this cell is already filled (should only be row 0)
        if grid[row][col] != -1:
            return solve_rc(row, col + 1)
            
        valid_vals = []
        for v in range(n):
            if not row_used[row][v] and not col_used[col][v]:
                ok = True
                for k in range(len(fixed_squares)):
                    s = fixed_squares[k][row][col]
                    if ortho_counts[k][s][v] > 0:
                        ok = False
                        break
                if ok:
                    valid_vals.append(v)
                    
        if not valid_vals:
            return False
            
        # Heuristic: MRV is not applied here, we just try all.
        # To improve, we could sort valid_vals by some heuristic.
        # Let's just shuffle.
        import random
        random.shuffle(valid_vals)
        
        for v in valid_vals:
            grid[row][col] = v
            row_used[row][v] = True
            col_used[col][v] = True
            for k in range(len(fixed_squares)):
                s = fixed_squares[k][row][col]
                ortho_counts[k][s][v] += 1
                
            if solve_rc(row, col + 1):
                return True
                
            grid[row][col] = -1
            row_used[row][v] = False
            col_used[col][v] = False
            for k in range(len(fixed_squares)):
                s = fixed_squares[k][row][col]
                ortho_counts[k][s][v] -= 1
                
        return False

    if solve_rc(1, 0):
        # Return a copy to ensure it's a list of lists of ints
        return [row[:] for row in grid]
    else:
        return None
