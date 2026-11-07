import time

def construct():
    n = 10
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
        # For each existing square k, for each symbol s_val, a bitmask of v's that have been paired with s_val
        # pair_used[k][s_val] is a bitmask where bit v is set if (s_val, v) has been used
        num_squares = len(existing_squares)
        pair_used = [[0] * n for _ in range(num_squares)]
        
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
                
                if backtrack_count(remaining - 1):
                    return True
                
                # Undo
                L_new[i][j] = -1
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k, s_val, bit_add in pair_to_add:
                    pair_used[k][s_val] &= ~bit_add
            
            return False

        success = backtrack_count(n * n)
        if success:
            return L_new
        else:
            return None
        """
        Finds a new Latin square orthogonal to all squares in existing_squares.
        Returns the square as a list of lists of ints, or None if not found within time_limit.
        """
        start_time = time.time()
        deadline = start_time + time_limit
        
        # State
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Constraints for L_new
        row_used = [0] * n  # Bitmask of symbols used in each row
        col_used = [0] * n  # Bitmask of symbols used in each col
        
        # Constraints for orthogonality with existing squares
        # pair_used[k] is a bitmask of used pairs (existing_squares[k][i][j], L_new[i][j])
        pair_used = [0] * len(existing_squares)
        
        def backtrack_count(remaining):
            if time.time() > deadline:
                return False
            if remaining == 0:
                return True
            
            min_count = n + 1
            best_i, best_j = -1, -1
            best_cands = []
            
            for i in range(n):
                for j in range(n):
                    if L_new[i][j] == -1:
                        cands = []
                        for v in range(n):
                            bit = 1 << v
                            if (row_used[i] & bit) or (col_used[j] & bit):
                                continue
                            
                            valid = True
                            for k, sq in enumerate(existing_squares):
                                s_val = sq[i][j]
                                pair_bit = 1 << (s_val * n + v)
                                if pair_used[k] & pair_bit:
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
                
                pair_bits = []
                for k, sq in enumerate(existing_squares):
                    s_val = sq[i][j]
                    pair_bit = 1 << (s_val * n + v)
                    pair_used[k] |= pair_bit
                    pair_bits.append(pair_bit)
                
                if backtrack_count(remaining - 1):
                    return True
                
                # Undo
                L_new[i][j] = -1
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k in range(len(existing_squares)):
                    pair_used[k] &= ~pair_bits[k]
            
            return False

        success = backtrack_count(n * n)
        if success:
            return L_new
        else:
            return None

    # Step 1: Find L2 orthogonal to L1
    # Time budget: 20s (leave more time for harder problems)
    L2 = find_orthogonal([L1], 20.0)
    
    if L2 is None:
        return [L1]
    
    squares = [L1, L2]
    
    # Step 2: Find L3 orthogonal to L1 and L2
    # Time budget: 30s
    L3 = find_orthogonal(squares, 30.0)
    
    if L3 is not None:
        squares.append(L3)
        
        # Step 3: Try to find L4 if time permits
        # Time budget: 20s
        L4 = find_orthogonal(squares, 20.0)
        if L4 is not None:
            squares.append(L4)
            
            # Step 4: Try to find L5 if time permits
            # Time budget: 10s
            L5 = find_orthogonal(squares, 10.0)
            if L5 is not None:
                squares.append(L5)
                
                # Step 5: Try to find L6 if time permits
                # Time budget: 5s
                L6 = find_orthogonal(squares, 5.0)
                if L6 is not None:
                    squares.append(L6)
                    
                    # Step 6: Try to find L7 if time permits
                    # Time budget: 3s
                    L7 = find_orthogonal(squares, 3.0)
                    if L7 is not None:
                        squares.append(L7)
                        
                        # Step 7: Try to find L8 if time permits
                        # Time budget: 2s
                        L8 = find_orthogonal(squares, 2.0)
                        if L8 is not None:
                            squares.append(L8)
                            
                            # Step 8: Try to find L9 if time permits
                            # Time budget: 1s
                            L9 = find_orthogonal(squares, 1.0)
                            if L9 is not None:
                                squares.append(L9)
    
    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
