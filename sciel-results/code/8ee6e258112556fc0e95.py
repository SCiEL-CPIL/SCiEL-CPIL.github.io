import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m
    rank = n * m
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # For each row i of A (and C), we use m columns of the rank decomposition
    for i in range(n):
        # Offset in rank columns for this i
        r_offset = i * m
        
        for j in range(m):
            r = r_offset + j
            
            # U[i*m + j, r] = 1
            U[i * m + j, r] = 1
            
            # V[j*p + k, r] = 1 for all k
            for k in range(p):
                V[j * p + k, r] = 1
            
            # W[k*n + i, r] = 1 for all k
            for k in range(p):
                W[k * n + i, r] = 1
                
    return {"U": U, "V": V, "W": W}
