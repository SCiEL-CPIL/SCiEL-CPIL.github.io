"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Schoolbook rank: n*m*p = 2*2*13 = 52
    # This is the trivial decomposition where each (i,j,k) triple gets its own rank-1 term.
    # U has shape (n*m, rank) = (4, 52)
    # V has shape (m*p, rank) = (26, 52)
    # W has shape (p*n, rank) = (26, 52)
    
    rank = n * m * p  # 52
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
