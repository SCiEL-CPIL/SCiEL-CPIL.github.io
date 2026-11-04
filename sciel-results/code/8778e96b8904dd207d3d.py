import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    n, m, p = 3, 4, 6
    
    # Trivial decomposition rank
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # r index: i * m * p + j * p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                # U[i, j] corresponds to row i*m + j
                U[i * m + j][r] = 1
                
                # V[j, k] corresponds to row j*p + k
                V[j * p + k][r] = 1
                
                # W[k, i] corresponds to row k*n + i
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
