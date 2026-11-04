import numpy as np
import time

def construct():
    n, m, p = 2, 4, 5
    rank = m  # 4
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # U[i*m + j, r] = delta_{j,r}
    for i in range(n):
        for j in range(m):
            U[i*m + j, j] = 1.0
    
    # V[j'*p + k, r] = delta_{j',r}
    for jp in range(m):
        for k in range(p):
            V[jp*p + k, jp] = 1.0
    
    # W[k*n + i, r] = 1 for all k, i, r
    for k in range(p):
        for i in range(n):
            for r in range(rank):
                W[k*n + i, r] = 1.0
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
