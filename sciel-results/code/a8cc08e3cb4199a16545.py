import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    # Rank R = n * m
    R = n * m
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for i in range(n):
        for k in range(m):
            r = i * m + k
            
            # U[i*m + k, r] = 1
            U[i * m + k][r] = 1
            
            # V[k*p + j, r] = 1 for all j in 0..p-1
            for j in range(p):
                V[k * p + j][r] = 1
            
            # W[j*n + i, r] = 1 for all j in 0..p-1
            # Note: W index is k*n + i in contract, where k is the output column index (j here)
            for j in range(p):
                W[j * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
