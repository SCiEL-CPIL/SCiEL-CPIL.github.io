import time
import random

def construct():
    n = 10
    L1_base = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal(existing_squares, time_limit):
        start_time = time.time()
        deadline = start_time + time_limit
        
        L_new = [[-1 for _ in range(n)] for _ in range(n)]
        row_used = [0] * n
        col_used = [0] * n
        num_squares = len(existing_squares)
        # pair_used[k][s_val] bitmask of v
        pair_used = [[0] * n for _ in range(num_squares)]
        
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

    # 1. Try cyclic
    L2 = find_orthogonal([L1_base], 2.0)
    if L2 is not None:
        return [L1_base, L2]
        
    # 2. Try randomized row permutations
    random.seed(12345)
    for _ in range(5):
        L_rand = [row[:] for row in L1_base]
        for i in range(n):
            random.shuffle(L_rand[i])
            
        L2_rand = find_orthogonal([L_rand], 10.0)
        if L2_rand is not None:
            return [L_rand, L2_rand]
            
    return [L1_base]
