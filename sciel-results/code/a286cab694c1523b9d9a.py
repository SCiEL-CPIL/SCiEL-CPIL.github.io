# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the schoolbook tensor decomposition for matrix multiplication.
    For matrices of size (n x m) and (m x p), the rank is n * m * p.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i (Trace convention: W[k*n+i, r])
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
