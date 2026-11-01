import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Naive rank-54 decomposition: one rank-1 term per (i,j,k) triple
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Term (i,j,k): U has 1 at (i*m+j), V has 1 at (j*p+k), W has 1 at (k*n+i)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
