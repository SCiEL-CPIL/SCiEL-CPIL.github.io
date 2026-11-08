import numpy as np
import time

def construct():
    n, m, p = 2, 3, 7
    
    rank = 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        r = j
        for i in range(n):
            U[i * m + j][r] = 1
        V[j * p + j][r] = 1
        for i in range(n):
            W[j * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
