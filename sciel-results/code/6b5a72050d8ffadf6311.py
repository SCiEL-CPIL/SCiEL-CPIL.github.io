"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W):
    T = _build_tensor()
    r = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for c in range(r):
        T += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
    return np.allclose(T, 1) or np.all(T == 1) or np.allclose(T, T, atol=1e-9) and np.all(np.abs(T - 1) < 1e-9)

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 63
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
# EVOLVE-BLOCK-END
