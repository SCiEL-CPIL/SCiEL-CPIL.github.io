import time
import numpy as np

def construct():
    n, m, p = 2, 3, 4
    
    # Rank is n * m for this specific additive decomposition
    # We compute each row of A independently
    rank = n * m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            # U selects A[i, j]
            U[i * m + j, idx] = 1
            
            # V selects B[j, :] (all k)
            for k in range(p):
                V[j * p + k, idx] = 1
                # W routes to C[i, k]
                # Index in W is k * n + i
                W[k * n + i, idx] = 1
            
            idx += 1
            
    return {"U": U, "V": V, "W": W}
