import time
import random

def construct():
    n = 10
    # Standard cyclic Latin square
    L1_base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal_fast(existing_squares, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        num_squares = len(existing_squares)
        
        # Pre-compute the values in existing squares for quick access
        # We need to check if (L1[i][j], L2[i][j]) is unique.
        # This is equivalent to: for each k in L1, the values in L2 at positions where L1==k must be distinct.
        
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Track usage
        row_used = [0] * n
        col_used = [0] * n
        # pair_used[k][s_val] is bitmask of values used in L_new at positions where existing_squares[k] has s_val
        # Wait, simpler: 
        # For each existing square S_k, and each symbol s, we track which values of L_new are used at positions (i,j) where S_k[i][j] == s.
        # Let's use a list of sets or bitmasks.
        # pair_used[k][s] = bitmask of L_new values used.
        pair_used = [[0] * n for _ in range(num_squares)]
        
        # List of empty positions
        empty_cells = [(i, j) for i in range(n) for j in range(n)]
        
        def get_valid_values(i, j):
            valid = []
            bit = 1
            for v in range(n):
                if not (row_used[i] & (1 << v)) and not (col_used[j] & (1 << v)):
                    ok = True
                    for k in range(num_squares):
                        s_val = existing_squares[k][i][j]
                        if pair_used[k][s_val] & (1 << v):
                            ok = False
                            break
                    if ok:
                        valid.append(v)
            return valid

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(empty_cells):
                return True
            
            i, j = empty_cells[idx]
            cands = get_valid_values(i, j)
            
            # Heuristic: shuffle candidates to randomize
            random.shuffle(cands)
            
            for v in cands:
                L_new[i][j] = v
                bit = 1 << v
                row_used[i] |= bit
                col_used[j] |= bit
                
                pair_changes = []
                for k in range(num_squares):
                    s_val = existing_squares[k][i][j]
                    pair_used[k][s_val] |= bit
                    pair_changes.append((k, s_val, bit))
                
                if backtrack(idx + 1):
                    return True
                
                # Undo
                L_new[i][j] = -1
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k, s_val, b in pair_changes:
                    pair_used[k][s_val] &= ~b
            
            return False

        # To make it faster, we can try to fill cells in a random order or use MRV.
        # MRV is better but requires scanning. For N=10, random order with backtracking is often fast enough if we restart.
        
        # Let's try a few restarts with different random orders
        for attempt in range(20):
            random.shuffle(empty_cells)
            # Reset state
            for i in range(n):
                for j in range(n):
                    L_new[i][j] = -1
            for i in range(n):
                row_used[i] = 0
                col_used[i] = 0
            for k in range(num_squares):
                for s in range(n):
                    pair_used[k][s] = 0
            
            if backtrack(0):
                return L_new
            
            if time.time() > deadline:
                break
                
        return None

    # 1. Try to find a second square
    L2 = find_orthogonal_fast([L1_base], 10.0)
    if L2 is not None:
        # Try to find a third square quickly
        L3 = find_orthogonal_fast([L1_base, L2], 5.0)
        if L3 is not None:
            return [L1_base, L2, L3]
        return [L1_base, L2]
        
    # If fast search fails, fall back to the old slower one or just return 1
    # But the fast one should work.
    
    return [L1_base]
