import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target tensor indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.append((a, b, c))
                
    def verify_solution(U, V, W, R):
        # U, V, W are lists of lists or numpy arrays
        # Check all valid indices sum to 1
        for a, b, c in valid_indices:
            val = 0.0
            for r in range(R):
                val += U[a][r] * V[b][r] * W[c][r]
            if abs(val - 1.0) > 1e-9:
                return False
        # Check some random invalid indices to ensure they are 0?
        # To be safe, check all invalid indices if R is small? 
        # There are nm*mp*pn - len(valid_indices) invalid indices.
        # For 2,3,4: 24*12*8 = 2304 total. 24 valid. 2280 invalid.
        # This is fast enough.
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) in valid_indices:
                        continue
                    val = 0.0
                    for r in range(R):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if abs(val) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Attempt to find rank < 24
    # We will try rank 23 and 22 with a few random integer restarts
    # This is much faster than gradient descent for exact integer solutions
    import random
    random.seed(42)
    
    max_time = 10 # seconds
    start_time = time.time()
    
    for R in [23, 22, 21]:
        if time.time() - start_time > max_time:
            break
            
        for _ in range(100): # 100 random restarts
            if time.time() - start_time > max_time:
                break
                
            # Random integer/half-integer initialization
            # We can just use small integers
            U = np.random.randint(-2, 3, (nm, R))
            V = np.random.randint(-2, 3, (mp, R))
            W = np.random.randint(-2, 3, (pn, R))
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_solution(U_list, V_list, W_list, R):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # Fallback to schoolbook
    return schoolbook()
