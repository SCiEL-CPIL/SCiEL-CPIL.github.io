# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Matrix multiplication 3x3 by 3x6.
    Returns a valid tensor decomposition.
    Currently uses the schoolbook algorithm (rank 54).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
