import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n  # 20, 30, 24
    
    # Schoolbook decomposition
    rank = n * m * p
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(np_)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                U[a][col] = 1.0
                V[b][col] = 1.0
                W[c][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
