# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 10
    # Construct first Latin square A
    A = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    def find_orthogonal(A):
        B = [[-1 for _ in range(n)] for _ in range(n)]
        row_masks = [0] * n
        col_masks = [0] * n
        pair_mask = 0
        
        def dfs(i, j):
            # Check deadline
            if time.time() > deadline:
                return False
                
            if i == n:
                return True
            if j == n:
                return dfs(i + 1, 0)
            
            if B[i][j] != -1:
                return dfs(i, j + 1)
            
            a_val = A[i][j]
            candidates = []
            for v in range(n):
                bit = 1 << v
                if not (row_masks[i] & bit) and not (col_masks[j] & bit):
                    pair_idx = a_val * 10 + v
                    if not (pair_mask & (1 << pair_idx)):
                        candidates.append(v)
                        
            if not candidates:
                return False
            
            # Shuffle candidates to randomize search path
            random.shuffle(candidates)
            
            for v in candidates:
                bit = 1 << v
                pair_idx = a_val * 10 + v
                bit_pair = 1 << pair_idx
                
                B[i][j] = v
                row_masks[i] |= bit
                col_masks[j] |= bit
                pair_mask |= bit_pair
                
                if dfs(i, j + 1):
                    return True
                
                B[i][j] = -1
                row_masks[i] ^= bit
                col_masks[j] ^= bit
                pair_mask ^= bit_pair
                
            return False
        
        # Try a few times with different random shuffles
        for _ in range(10):
            if time.time() > deadline:
                break
            if dfs(0, 0):
                return B
        return None

    B = find_orthogonal(A)
    
    if B:
        # Ensure types are Python int
        A = [[int(x) for x in row] for row in A]
        B = [[int(x) for x in row] for row in B]
        return [A, B]
    else:
        A = [[int(x) for x in row] for row in A]
        return [A]
# EVOLVE-BLOCK-END
