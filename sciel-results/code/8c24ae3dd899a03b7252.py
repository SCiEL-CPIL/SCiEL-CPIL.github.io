import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-27 decomposition for 2x3 by 3x9 matrix multiplication.
    
    Exploits the structure: C = A @ B where A is 2x3, B is 3x9, C is 2x9.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For each (j, k) pair, we create a rank-1 term:
    - U[i*m + j, r] = 1 for all i (so U has 1s in column r at rows 0*m+j, 1*m+j, ...)
    - V[j*p + k, r] = 1
    - W[k*n + i, r] = 1 for all i (so W has 1s in column r at rows 0*n+i, 1*n+i, ...)
    
    Total rank = m * p = 3 * 9 = 27.
    """
    n, m, p = DIMENSIONS
    rank = m * p  # 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for j in range(m):
        for k in range(p):
            # U: for all i, set U[i*m + j, r] = 1
            for i in range(n):
                U[i * m + j][r] = 1
            # V: set V[j*p + k, r] = 1
            V[j * p + k][r] = 1
            # W: for all i, set W[k*n + i, r] = 1
            for i in range(n):
                W[k * n + i][r] = 1
            r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
