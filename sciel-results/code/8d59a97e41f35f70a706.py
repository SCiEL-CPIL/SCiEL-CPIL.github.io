"""Safe exact-rational schoolbook for 4x4x7 with deadline guard."""
import time
from fractions import Fraction as F

DIMENSIONS = [4, 4, 7]
_START = time.time()
_DEADLINE = _START + 140.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = F(1)
                V[j * p + k][col] = F(1)
                W[k * n + i][col] = F(1)
                col += 1
    return {"U": U, "V": V, "W": W}
