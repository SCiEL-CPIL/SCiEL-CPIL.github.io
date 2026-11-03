"""Program for matrix multiplication 2x4x5 tensor decomposition."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
_DEADLINE = None

def _start_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def construct():
    _start_deadline()
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p.
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
# EVOLVE-BLOCK-END
