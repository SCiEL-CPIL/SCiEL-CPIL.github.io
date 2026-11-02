import numpy as np
import itertools

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    for i, j, k in itertools.product(range(n), range(m), range(p)):
        U[i * m + j][idx] = 1
        V[j * p + k][idx] = 1
        W[k * n + i][idx] = 1
        idx += 1
        
    return {"U": U, "V": V, "W": W}
