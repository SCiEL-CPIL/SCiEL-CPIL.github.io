import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 52 decomposition
    # Part 1: 4x4x4 (16)
    # Part 2: 4x4x1 (16)
    # Part 3: 4x1x5 (20)
    
    rank_total = 52
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col = 0
    
    # Part 1
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Part 2
    for i in range(4):
        for j in range(4):
            k = 4
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    # Part 3
    for i in range(4):
        j = 4
        for k in range(5):
            U[i*m + j, col] = 1
            V[j*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
