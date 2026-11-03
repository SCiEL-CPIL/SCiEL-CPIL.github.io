import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank m solution (summing over j)
    R = m # 3
    
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # r maps to j
    for j in range(m):
        r = j
        
        # U[i*m + j, r] = 1 for all i
        for i in range(n):
            U[i * m + j][r] = 1
            
        # V[j*p + k, r] = 1 for all k
        for k in range(p):
            V[j * p + k][r] = 1
            
        # W[k*n + i, r] = 1 for all i, k
        for i in range(n):
            for k in range(p):
                W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
