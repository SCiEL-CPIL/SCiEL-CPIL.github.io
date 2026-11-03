"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 25
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # U[i*m + j, r] = 1 for all i
            for i in range(n):
                U[i * m + j, r] = 1.0
            
            # V[j*p + k, r] = 1
            V[j * p + k, r] = 1.0
            
            # W[k*n + i, r] = 1 for all i
            for i in range(n):
                W[k * n + i, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
