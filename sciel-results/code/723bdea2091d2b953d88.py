"""Matrix multiplication 4x4 by 4x9: schoolbook tensor decomposition."""
import time

DIMENSIONS = [4, 4, 9]
_START = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication
    of dimensions n x m * m x p.
    
    The schoolbook rank is n * m * p.
    For 4x4 * 4x9, rank = 144.
    
    The decomposition satisfies:
    sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for all valid indices i, j, k.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
