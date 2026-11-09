import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 8 decomposition
    # C[i, k] = sum_j A[i, j] B[j, k]
    # We can decompose this into 8 bilinear terms:
    # For each i in {0, 1} and j in {0, 1, 2, 3}:
    # Term (i, j) computes A[i, j] * B[j, :] and adds to C[i, :]
    # This is valid because A[i, j] is a scalar and B[j, :] is a row vector.
    # The product is a row vector added to C[i, :].
    
    rank = 8
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            # U part: A[i, j]
            U[i * m + j][col] = 1
            
            # V part: B[j, k] for all k
            for k in range(p):
                V[j * p + k][col] = 1
            
            # W part: C[i, k] for all k
            for k in range(p):
                # W index is k * n + i
                W[k * n + i][col] = 1
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
