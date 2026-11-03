"""Matrix multiplication 2,3,9 tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _schoolbook_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
    rank = n * m * p
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
    return U, V, W

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses the schoolbook decomposition which is exact and guaranteed to finish quickly.
    """
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
