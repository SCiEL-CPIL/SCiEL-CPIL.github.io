import time
import sys
import random

def construct():
    n = 10
    start_time = time.time()
    deadline = start_time + 50.0  # 10s margin
    
    # 1. Create the first Latin Square (Cyclic)
    S0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def is_valid_latin(sq):
        for i in range(n):
            if set(sq[i]) != set(range(n)):
                return False
            for j in range(n):
                col = [sq[k][j] for k in range(n)]
                if set(col) != set(range(n)):
                    return False
        return True

    def check_orthogonality(sq_a, sq_b):
        seen = set()
        for i in range(n):
            for j in range(n):
                p = (sq_a[i][j], sq_b[i][j])
                if p in seen:
                    return False
                seen.add(p)
        return True

    # Backtracking solver for a new square orthogonal to a list of existing squares
    def find_orthogonal_latin(existing_squares):
        # existing_squares is a list of lists of lists
        sq = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Pre-compute constraints from existing squares
        # For each cell (i,j), the value in sq[i][j] cannot form a duplicate pair with existing squares
        # Also must satisfy Latin constraints.
        
        # We will fill cells in a fixed order, e.g., row by row.
        # To improve, we can choose the cell with fewest candidates (MRV).
        # But for simplicity and speed in Python, fixed order with strong pruning might suffice for N=10.
        
        row_masks = [0] * n  # bitmask of used symbols in row
        col_masks = [0] * n  # bitmask of used symbols in col
        pair_masks = [0] * (len(existing_squares)) # bitmask of used pairs? 
        # Pair masks are 2D: for each existing square k, we track which (val_k, val_new) pairs are used.
        # Actually, we need to ensure (val_k, val_new) is unique for each k.
        # So for each k, we have a set of used pairs.
        # Using sets is slower than bitmasks. But 100 pairs is small.
        
        used_pairs = [set() for _ in existing_squares]
        
        def is_consistent(i, j, val):
            # Check Latin
            if (row_masks[i] >> val) & 1:
                return False
            if (col_masks[j] >> val) & 1:
                return False
            
            # Check Orthogonality
            for k, ex_sq in enumerate(existing_squares):
                ex_val = ex_sq[i][j]
                pair = (ex_val, val)
                if pair in used_pairs[k]:
                    return False
            return True

        def backtrack(i, j):
            if time.time() > deadline:
                return False
            
            if i == n:
                return True
            
            ni = i
            nj = j + 1
            if nj == n:
                ni = i + 1
                nj = 0
            
            # MRV: Try to find a cell in the remaining unassigned cells with fewest options?
            # For now, simple order. If too slow, optimize.
            
            # Determine candidates for (i, j)
            candidates = []
            for val in range(n):
                if is_consistent(i, j, val):
                    candidates.append(val)
            
            if not candidates:
                return False
            
            # Shuffle candidates to break symmetry / randomize search path?
            # Randomization helps avoid getting stuck in one bad branch.
            random.shuffle(candidates)
            
            for val in candidates:
                # Assign
                sq[i][j] = val
                row_masks[i] |= (1 << val)
                col_masks[j] |= (1 << val)
                for k in range(len(existing_squares)):
                    used_pairs[k].add((existing_squares[k][i][j], val))
                
                if backtrack(ni, nj):
                    return True
                
                # Unassign
                sq[i][j] = -1
                row_masks[i] &= ~(1 << val)
                col_masks[j] &= ~(1 << val)
                for k in range(len(existing_squares)):
                    used_pairs[k].discard((existing_squares[k][i][j], val))
            
            return False

        # Start backtracking
        if backtrack(0, 0):
            return sq
        else:
            return None

    # Try to find S1
    S1 = find_orthogonal_latin([S0])
    
    result = [S0]
    if S1:
        result.append(S1)
        
    # Try to find S2
    if time.time() < deadline:
        S2 = find_orthogonal_latin([S0, S1])
        if S2:
            result.append(S2)
            
    # Final validation
    valid_result = []
    for sq in result:
        if not is_valid_latin(sq):
            continue
        ok = True
        for v_sq in valid_result:
            if not check_orthogonality(sq, v_sq):
                ok = False
                break
        if ok:
            valid_result.append(sq)
            
    if not valid_result:
        return [S0]
        
    return valid_result
