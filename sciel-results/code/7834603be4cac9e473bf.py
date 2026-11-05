import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the rank-60 schoolbook decomposition
    # This is the exact, trivial decomposition where each term A[i,j] * B[j,k]
    # is handled by a separate rank-1 tensor.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][r] = 1
                # V index: j*p + k
                V[j * p + k][r] = 1
                # W index: k*n + i
                W[k * n + i][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
