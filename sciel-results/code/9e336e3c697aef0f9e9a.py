"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # L1: the standard cyclic Latin square
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal(existing_squares, time_limit):
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
    # Time budget: 25s
    L2 = find_orthogonal([L1], 25.0)
    
    if L2 is None:
        return [L1]
    
    squares = [L1, L2]
    
    # Step 2: Find L3 orthogonal to L1 and L2
    # Time budget: 50s (leaves 15s margin for overhead/return)
    L3 = find_orthogonal(squares, 50.0)
    
    if L3 is not None:
        squares.append(L3)
        
    # Ensure types
    final_result = []
    for sq in squares:
        final_sq = []
        for row in sq:
            final_row = [int(x) for x in row]
            final_sq.append(final_row)
        final_result.append(final_sq)
        
    return final_result

# EVOLVE-BLOCK-END
