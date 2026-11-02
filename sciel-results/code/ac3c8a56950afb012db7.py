import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                
                r += 1
                
    # Verify the decomposition
    # Check that sum_r U[i*m+jp, r] * V[jm*p+k, r] * W[k*n+i, r] = delta_{jp,jm}
    max_err = 0.0
    for i in range(n):
        for k in range(p):
            for jp in range(m):
                for jm in range(m):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+jp, r] * V[jm*p+k, r] * W[k*n+i, r]
                    target = 1.0 if jp == jm else 0.0
                    max_err = max(max_err, abs(val - target))
    
    if max_err > 1e-10:
        raise ValueError(f"Decomposition verification failed with max error {max_err}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
