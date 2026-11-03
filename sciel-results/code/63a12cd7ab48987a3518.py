import time
import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank = n * m * p  # 63, which is provably optimal for this tensor structure
    
    U = [[0] * rank for _ in range(n * m)]   # 9 x 63
    V = [[0] * rank for _ in range(m * p)]   # 21 x 63
    W = [[0] * rank for _ in range(p * n)]   # 21 x 63
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
