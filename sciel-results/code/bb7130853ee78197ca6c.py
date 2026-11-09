"""Tensor decomposition for 3x4x4 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 4]
_DEADLINE = 150.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - 3.0 - _elapsed()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 48
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
# EVOLVE-BLOCK-END
