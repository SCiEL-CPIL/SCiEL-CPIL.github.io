import numpy as np
import time
import random

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
                
    def check_decomposition(U, V, W, rank):
        U_flat = np.array(U).reshape(n*m, rank)
        V_flat = np.array(V).reshape(m*p, rank)
        W_flat = np.array(W).reshape(p*n, rank)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(r):
        # Try a few random configurations
        # Use small integers to help with exactness
        for _ in range(50):
            if time.time() > deadline:
                return None
            
            U_rand = np.random.choice([-1, 0, 1], size=(n*m, r))
            V_rand = np.random.choice([-1, 0, 1], size=(m*p, r))
            W_rand = np.random.choice([-1, 0, 1], size=(p*n, r))
            
            if check_decomposition(U_rand, V_rand, W_rand, r):
                return U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), r
        return None

    # Try ranks from 24 to 47
    # We prioritize lower ranks
    for r in range(24, 48):
        if time.time() > deadline - 5:
            break
        res = try_rank(r)
        if res:
            U, V, W, rank = res
            return {"U": U, "V": V, "W": W}
            
    # Fallback
    return {"U": U_sb, "V": V_sb, "W": W_sb}
