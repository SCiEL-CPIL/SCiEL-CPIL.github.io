import time

def construct():
    n = 10
    # L1 is the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    start_time = time.time()
    # Keep a safety margin from the 90s wall-clock limit
    deadline = 85.0 

    def find_next_square(existing_squares, time_limit):
        """
        Finds a new Latin square orthogonal to all squares in existing_squares.
        """
        if time.time() > deadline:
            return None
            
        # State
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Constraints for L_new
        row_used = [0] * n
        col_used = [0] * n
        
        # Constraints for orthogonality with existing squares
        pair_used = [0] * len(existing_squares)
        
        # Precompute pair bits
        pair_bit_cache = [[[] for _ in range(n)] for _ in range(len(existing_squares))]
        for k, sq in enumerate(existing_squares):
            for i in range(n):
                for j in range(n):
                    s_val = sq[i][j]
                    pair_bit_cache[k][i][j] = [1 << (s_val * n + v) for v in range(n)]
        
        def get_candidates(i, j):
            cands = []
            for v in range(n):
                bit = 1 << v
                if (row_used[i] & bit) or (col_used[j] & bit):
                    continue
                valid = True
                for k in range(len(existing_squares)):
                    if pair_used[k] & pair_bit_cache[k][i][j][v]:
                        valid = False
                        break
                if valid:
                    cands.append(v)
            return cands
        
        def backtrack(remaining):
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
                        cands = get_candidates(i, j)
                        count = len(cands)
                        if count == 0:
                            return False
                        if count < min_count:
                            min_count = count
                            best_i, best_j = i, j
                            best_cands = cands
                            if count == 1:
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
                for k in range(len(existing_squares)):
                    pb = pair_bit_cache[k][i][j][v]
                    pair_used[k] |= pb
                    pair_bits.append(pb)
                
                if backtrack(remaining - 1):
                    return True
                
                L_new[i][j] = -1
                row_used[i] &= ~bit
                col_used[j] &= ~bit
                for k in range(len(existing_squares)):
                    pair_used[k] &= ~pair_bits[k]
            
            return False

        success = backtrack(n * n)
        if success:
            return L_new
        else:
            return None

    squares = [L1]
    
    # Find L2
    if time.time() < deadline - 5:
        L2 = find_next_square(squares, deadline - time.time())
        if L2:
            squares.append(L2)
            
    # Find L3 (will likely fail, but we try)
    if len(squares) >= 2 and time.time() < deadline - 5:
        L3 = find_next_square(squares, deadline - time.time())
        if L3:
            squares.append(L3)
            
    # Find L4
    if len(squares) >= 3 and time.time() < deadline - 5:
        L4 = find_next_square(squares, deadline - time.time())
        if L4:
            squares.append(L4)

    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
