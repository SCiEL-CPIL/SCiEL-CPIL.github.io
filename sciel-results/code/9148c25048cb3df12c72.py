import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 12
    dim_V = m * p  # 28
    dim_W = p * n  # 21
    
    rank = dim_V  # 28
    
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    # Construction:
    # For each column c in 0..rank-1:
    # Map c to (j, k) such that c = j * p + k
    # j in 0..m-1, k in 0..p-1
    # Set V[j * p + k, c] = 1
    # Set U[i * m + j, c] = 1 for all i in 0..n-1
    # Set W[k * n + i, c] = 1 for all i in 0..n-1
    
    for c in range(rank):
        j = c // p
        k = c % p
        
        V[j * p + k, c] = 1
        
        for i in range(n):
            U[i * m + j, c] = 1
            W[k * n + i, c] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
