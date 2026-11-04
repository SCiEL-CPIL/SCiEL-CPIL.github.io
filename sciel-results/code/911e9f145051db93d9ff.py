import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n = 4
    m = 4
    p = 8
    
    # Rank for naive multiplication
    R = n * m * p
    
    # Initialize matrices with zeros
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    # Populate matrices for each (i, j, k) term
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                u_idx = i * m + j
                U[u_idx, r] = 1.0
                
                v_idx = j * p + k
                V[v_idx, r] = 1.0
                
                w_idx = k * n + i
                W[w_idx, r] = 1.0
                
                r += 1
                
    # If we broke early, we need to return a valid partial result?
    # The task requires an *exact* decomposition.
    # If we break early, it's not exact.
    # But 128 iterations is very fast.
    
    return {'U': U, 'V': V, 'W': W}
