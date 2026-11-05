"""Matrix multiplication 4x5x5 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    
    Uses the standard schoolbook algorithm, which guarantees an exact decomposition
    with rank n*m*p = 4*5*5 = 100.
    """
    n, m, p = 4, 5, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j, t] = 1
                # V index: j * p + k
                V[j * p + k, t] = 1
                # W index: k * n + i (trace convention W[k*n+i, r])
                W[k * n + i, t] = 1
                t += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
