import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Target rank: 39 (27 for 3x3x3 core + 12 for 3x3x4 tail)
    rank = 39
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: 3x3x3 Core (Schoolbook Rank 27)
    # For each i, j, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    # Part 2: 3x3x4 Tail (Rank 12)
    # For each j in 0..2, k in 3..6
    # We use a factor that sums over i
    for j in range(m):
        for k in range(3, p):
            for i in range(n):
                U[i*m + j, col] = 1.0
            V[j*p + k, col] = 1.0
            for i in range(n):
                W[k*n + i, col] = 1.0
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
