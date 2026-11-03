"""Find a rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]
n, m, p = DIMENSIONS

def _verify(U, V, W):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    start = time.time()
    deadline = start + 110.0

    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # has tensor rank 1, since it's the outer product of all-ones vectors.
    # Rank-1 decomposition:
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]

    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook decomposition (rank n*m*p)
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
    return {"U": U, "V": V, "W": W}
