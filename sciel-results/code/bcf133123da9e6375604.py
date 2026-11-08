"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the factor matrices for the tensor decomposition of 3x4 by 4x4 matrix multiplication.
    Uses the standard bilinear algorithm which corresponds to the rank m*n*p decomposition.
    For 3x4*4x4, rank = 3*4*4 = 48.
    
    Indexing conventions:
    A (n x m): 3x4. Index u = i*m + j (row-major).
    B (m x p): 4x4. Index v = j*p + k (row-major).
    C (n x p): 3x4. Index w = k*n + i (column-major, as per prompt W[k*n+i, r]).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                u = i * m + j
                v = j * p + k
                w = k * n + i
                
                U[u, r] = 1.0
                V[v, r] = 1.0
                W[w, r] = 1.0
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
