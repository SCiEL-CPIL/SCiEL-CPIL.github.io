import numpy as np
import time

def construct():
    n, m, p = 4, 5, 5
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A11 * B11 (4x4 * 4x4)
    # i in 0..3, j in 0..3, k in 0..3
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
                
    # Part 2: A12 * B21 (4x1 * 1x4)
    # i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            U[i * m + 4][r] = 1
            V[4 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 3: A11 * B12 (4x4 * 4x1)
    # i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 4: A12 * B22 (4x1 * 1x1)
    # i in 0..3, j=4, k=4
    for i in range(4):
        U[i * m + 4][r] = 1
        V[4 * p + 4][r] = 1
        W[4 * n + i][r] = 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
