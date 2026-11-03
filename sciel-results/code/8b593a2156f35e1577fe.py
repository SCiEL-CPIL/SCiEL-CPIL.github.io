import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = 2, 5, 5
    rank = n * m * p
    
    # U shape: (n*m, rank) = (10, 50)
    # V shape: (m*p, rank) = (25, 50)
    # W shape: (p*n, rank) = (10, 50)
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        if time.time() > deadline:
            break
        
        # Map r to i, j, k
        # r = i * m * p + j * p + k
        k = r % p
        j = (r // p) % m
        i = (r // (p * m)) % n
        
        # U index: i * m + j
        u_idx = i * m + j
        U[u_idx, r] = 1
        
        # V index: j * p + k
        v_idx = j * p + k
        V[v_idx, r] = 1
        
        # W index: k * n + i
        w_idx = k * n + i
        W[w_idx, r] = 1
        
    # If we broke early, the result is invalid/incomplete.
    # But rank 50 loop is very fast (50 iterations).
    # This guard is for compliance with the directive.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
