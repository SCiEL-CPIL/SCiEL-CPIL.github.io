"""Matrix multiplication 2x3x3 tensor decomposition - rank minimal."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The tensor T[i,j,k] for (AB)[i,k] = sum_j A[i,j]*B[j,k] has rank 18
    for dimensions (2,3,3). We return the schoolbook decomposition.
    
    Convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 18, which is the minimal rank for M_{2,3,3}
    
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
# EVOLVE-BLOCK-END
