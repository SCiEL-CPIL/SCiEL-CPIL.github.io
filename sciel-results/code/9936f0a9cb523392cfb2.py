import time
import sys

def construct():
    n = 10
    # L1 is the standard Latin square: L1[i][j] = (i + j) % n
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal(existing_squares, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        
        # Fix the first row to 0, 1, ..., n-1
        for j in range(n):
            L_new[0][j] = j
            
        row_used = [0] * n
        col_used = [0] * n
        
        # Initialize row_used and col_used for the first row
        row_used[0] = (1 << n) - 1
        for j in range(n):
            col_used[j] |= (1 << j)
            
        # Constraints for orthogonality
        # pair_used[k] is a bitmask of used pairs (existing_squares[k][i][j], L_new[i][j])
        # Since n=10, pair index is s_val * 10 + v. Max index 99.
        # 100 bits fit in a Python int.
        pair_used = [0] * len(existing_squares)
        
        # Initialize pair_used for the first row
        for k, sq in enumerate(existing_squares):
            for j in range(n):
                s_val = sq[0][j]
                v = j # L_new[0][j] is j
                pair_bit = 1 << (s_val * n + v)
                pair_used[k] |= pair_bit
        
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

        success = backtrack_count(n * n - n) # n cells in first row are filled
        if success:
            return L_new
        else:
            return None

    squares = [L1]
    
    # Try to find L2
    L2 = find_orthogonal(squares, 20.0)
    if L2 is not None:
        squares.append(L2)
        
    # Try to find L3
    if len(squares) >= 2:
        L3 = find_orthogonal(squares, 40.0)
        if L3 is not None:
            squares.append(L3)
            
    # Try to find L4
    if len(squares) >= 3:
        L4 = find_orthogonal(squares, 55.0)
        if L4 is not None:
            squares.append(L4)

    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result
