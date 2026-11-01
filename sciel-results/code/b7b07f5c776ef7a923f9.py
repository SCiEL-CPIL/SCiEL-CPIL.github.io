import time
import sys

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Returns the schoolbook tensor decomposition for multiplying a 2x3 matrix
    by a 3x6 matrix. The rank is n*m*p = 36.

    U: (n*m, rank) = (6, 36)
    V: (m*p, rank) = (18, 36)
    W: (p*n, rank) = (12, 36)

    Each rank-1 term corresponds to a unique (i, j, k) triplet where
    i in [0, n), j in [0, m), k in [0, p).
    """
    n, m, p = DIMENSIONS
    rank = n * m * p

    # Build U, V, W as lists of lists of integers
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
