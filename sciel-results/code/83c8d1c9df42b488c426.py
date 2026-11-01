import time
import sys

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # Rank is p * m = 9 * 3 = 27
    rank = p * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for k in range(p):
        for j in range(m):
            # U part: rows 0 and 1, column j
            U[0 * m + j][r] = 1
            U[1 * m + j][r] = 1
            
            # V part: row j, column k
            V[j * p + k][r] = 1
            
            # W part: rows 0 and 1, column k
            # W index is k * n + i
            W[k * n + 0][r] = 1
            W[k * n + 1][r] = 1
            
            r += 1
            
    return {"U": U, "V": V, "W": W}
