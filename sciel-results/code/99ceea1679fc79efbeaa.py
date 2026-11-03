import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = 1
    U = [[1.0] * rank for _ in range(n * m)]
    V = [[1.0] * rank for _ in range(m * p)]
    W = [[1.0] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
