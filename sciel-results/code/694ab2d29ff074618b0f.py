"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                expected = 1.0
                if abs(val - expected) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    rank = n * m * p
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
    
    # Verify the schoolbook decomposition
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
        raise RuntimeError("Schoolbook decomposition failed verification")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
