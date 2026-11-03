import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook algorithm for matrix multiplication
    # Rank = n * m
    rank = n * m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # r corresponds to (i0, j0)
    for i0 in range(n):
        for j0 in range(m):
            r = i0 * m + j0
            
            # U[i*m + j, r] = 1 if i==i0 and j==j0
            u_idx = i0 * m + j0
            U[u_idx, r] = 1
            
            # V[j*p + k, r] = 1 if j==j0 (for all k)
            for k in range(p):
                v_idx = j0 * p + k
                V[v_idx, r] = 1
            
            # W[k*n + i, r] = 1 if i==i0 (for all k)
            for k in range(p):
                w_idx = k * n + i0
                W[w_idx, r] = 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
