import time

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal(existing_squares, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        
        # State
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Constraints for L_new
        row_used = [0] * n  # Bitmask of symbols used in each row
        col_used = [0] * n  # Bitmask of symbols used in each col
        
        # Constraints for orthogonality with existing squares
        num_squares = len(existing_squares)
        pair_used = [[0] * n for _ in range(num_squares)]
        
        # Symmetry Breaking: Fix the first row to 0, 1, ..., n-1
        for j in range(n):
            v = j
            L_new[0][j] = v
            bit = 1 << v
            row_used[0] |= bit
            col_used[j] |= bit
            
            for k in range(num_squares):
                s_val = existing_squares[k][0][j]
                pair_used[k][s_val] |= bit
        
        remaining = n * n - n # 90 cells left
        
        # Helper to check if a cell (i, j) has any valid candidates
        # Returns True if valid candidates exist, False otherwise
        def has_valid_candidates(i, j):
            if L_new[i][j] != -1:
                return True
            for v in range(n):
                bit = 1 << v
                if (row_used[i] & bit) or (col_used[j] & bit):
                    continue
                valid = True
                for k in range(num_squares):
                    s_val = existing_squares[k][i][j]
                    if pair_used[k][s_val] & bit:
                        valid = False
                        break
                if valid:
                    return True
            return False

        def backtrack_count(remaining):
            if time.time() > deadline:
                return False
            if remaining == 0:
                return True
            
            # Find cell with minimum remaining values (MRV)
            min_count = n + 1
            best_i, best_j = -1, -1
            best_cands = []
            
            for i in range(n):
                if i == 0: continue
                for j in range(n):
                    if L_new[i][j] == -1:
                        cands = []
                        for v in range(n):
                            bit = 1 << v
                            if (row_used[i] & bit) or (col_used[j] & bit):
                                continue
                            
                            valid = True
                            for k in range(num_squares):
                                s_val = existing_squares[k][i][j]
                                if pair_used[k][s_val] & bit:
                                    valid = False
                                    break
                            
                            if valid:
                                cands.append(v)
                        
                        if len(cands) < min_count:
                            min_count = len(cands)
                            best_i, best_j = i, j
                            best_cands = cands
                            if min_count == 0:
                                return False
                            if min_count == 1:
                                break
                if min_count == 1:
                    break
            
            if best_i == -1:
                return True
            
            i, j = best_i, best_j
            
            for v in best_cands:
                L_new[i][j] = v
                bit = 1 << v
                row_used[i] |= bit
                col_used[j] |= bit
                
                pair_to_add = []
                for k in range(num_squares):
                    s_val = existing_squares[k][i][j]
                    pair_used[k][s_val] |= bit
                    pair_to_add.append((k, s_val, bit))
                
                # Forward Checking
                feasible = True
                if remaining > 1: # Only check if there are other empty cells
                    for r in range(n):
                        if r == 0: continue
                        for c in range(n):
                            if L_new[r][c] == -1 and not has_valid_candidates(r, c):
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    if backtrack_count(remaining - 1):
                        return True
                
                # Undo
                L_new[i][j] = -1
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k, s_val, bit_add in pair_to_add:
                    pair_used[k][s_val] &= ~bit_add
            
            return False

        success = backtrack_count(remaining)
        if success:
            return L_new
        else:
            return None

    squares = [L1]
    
    start_total = time.time()
    total_limit = 85.0 # Safety margin
    
    # Dynamic time allocation
    # L2 is usually fast.
    # If L2 is found, try L3 with remaining time.
    
    # Try L2
    t_start = time.time()
    L2 = find_orthogonal(squares, 5.0)
    t_elapsed = time.time() - t_start
    if L2 is not None:
        squares.append(L2)
    
    # Try L3
    elapsed_total = time.time() - start_total
    if len(squares) == 2 and elapsed_total < total_limit - 5.0:
        # Give L3 a significant chunk of the remaining time
        # If L2 was fast, we have lots of time.
        # Let's say we reserve 5s for potential L4, L5...
        # But L3 is the hardest bottleneck usually.
        budget_L3 = min(40.0, total_limit - elapsed_total - 10.0)
        if budget_L3 > 2.0:
            L3 = find_orthogonal(squares, budget_L3)
            if L3 is not None:
                squares.append(L3)
                
                # Try L4
                elapsed_total = time.time() - start_total
                if len(squares) == 3 and elapsed_total < total_limit - 5.0:
                    budget_L4 = min(20.0, total_limit - elapsed_total - 5.0)
                    if budget_L4 > 1.0:
                        L4 = find_orthogonal(squares, budget_L4)
                        if L4 is not None:
                            squares.append(L4)
                            
                            # Try L5
                            elapsed_total = time.time() - start_total
                            if len(squares) == 4 and elapsed_total < total_limit - 2.0:
                                budget_L5 = min(10.0, total_limit - elapsed_total - 1.0)
                                if budget_L5 > 0.5:
                                    L5 = find_orthogonal(squares, budget_L5)
                                    if L5 is not None:
                                        squares.append(L5)

    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
