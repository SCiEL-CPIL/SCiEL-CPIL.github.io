import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 12
    np_ = n * p  # 8
    
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(np_)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
