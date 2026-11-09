import random
import time

def construct():
    n = 10
    start_time = time.time()
    timeout = 55.0 # Leave some buffer
    
    # Generate first square (cyclic)
    squares = []
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares.append(sq0)
    
    # Try to find more squares
    for _ in range(8): # Max 9 total
        if time.time() - start_time > timeout:
            break
        
        # Backtracking to find next square
        next_sq = find_next_square(n, squares)
        if next_sq is not None:
            squares.append(next_sq)
    
    return squares

def find_next_square(n, existing_squares):
    # State variables
    grid = [[-1 for _ in range(n)] for _ in range(n)]
    row_masks = [0] * n
    col_masks = [0] * n
    
    # pair_used[k][v_new] is a bitmask of v_k values that are already paired with v_new
    # existing_squares has length K
    pair_used = [[0] * n for _ in range(len(existing_squares))]
    
    # Pre-compute values of existing squares for faster access?
    # sq_vals[k][i][j] = existing_squares[k][i][j]
    sq_vals = existing_squares
    
    def count_candidates(i, j):
        """Count valid candidates for cell (i, j)"""
        if grid[i][j] != -1:
            return 0
        count = 0
        # Pre-compute the bitmask of available values for this row/col
        # Actually, just iterate 0..n-1
        for v in range(n):
            bit = 1 << v
            if (row_masks[i] & bit) or (col_masks[j] & bit):
                continue
            
            valid = True
            for k in range(len(sq_vals)):
                v_k = sq_vals[k][i][j]
                if (pair_used[k][v] & (1 << v_k)):
                    valid = False
                    break
            
            if valid:
                count += 1
        return count

    def get_candidates(i, j):
        """Get list of valid candidates for cell (i, j)"""
        if grid[i][j] != -1:
            return []
        candidates = []
        for v in range(n):
            bit = 1 << v
            if (row_masks[i] & bit) or (col_masks[j] & bit):
                continue
            
            valid = True
            for k in range(len(sq_vals)):
                v_k = sq_vals[k][i][j]
                if (pair_used[k][v] & (1 << v_k)):
                    valid = False
                    break
            
            if valid:
                candidates.append(v)
        return candidates

    def backtrack():
        if time.time() > 55.0:
            return False
        
        # Find cell with minimum remaining values (MRV)
        min_count = n + 1
        best_i, best_j = -1, -1
        best_cands = []
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] == -1:
                    cands = get_candidates(i, j)
                    if len(cands) < min_count:
                        min_count = len(cands)
                        best_i, best_j = i, j
                        best_cands = cands
                        if min_count == 0:
                            return False
                        if min_count == 1:
                            # Can't get better than 1
                            break
            if min_count == 1:
                break
        
        if best_i == -1:
            return True # All cells filled
        
        # Try each candidate
        for v in best_cands:
            # Apply
            grid[best_i][best_j] = v
            bit = 1 << v
            row_masks[best_i] |= bit
            col_masks[best_j] |= bit
            
            for k in range(len(sq_vals)):
                v_k = sq_vals[k][best_i][best_j]
                pair_used[k][v] |= (1 << v_k)
            
            # Check if this placement causes any immediate contradiction in the same row/col
            # Specifically, if any other cell in row best_i or col best_j has 0 candidates
            # We only need to check cells that were affected, i.e., in the same row or col
            # But wait, pair_used affects ALL cells.
            # However, row/col masks only affect same row/col.
            # So cells outside row best_i and col best_j are only affected by pair_used.
            # Checking ALL cells is expensive.
            # Let's just recurse. The MRV step in the next call will detect if any cell has 0 cands.
            # To optimize, we can check only the row and col of the placed cell for 0-cands.
            # If a cell elsewhere becomes 0-cands due to pair_used, it will be caught in the next MRV step.
            # Is it worth checking? 
            # Let's try just recursing first. If it's too slow, we add forward checking.
            
            if backtrack():
                return True
            
            # Undo
            grid[best_i][best_j] = -1
            row_masks[best_i] &= ~bit
            col_masks[best_j] &= ~bit
            for k in range(len(sq_vals)):
                v_k = sq_vals[k][best_i][best_j]
                pair_used[k][v] &= ~(1 << v_k)
        
        return False

    if backtrack():
        return grid
    else:
        return None
