import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        base = t * 7
        for s in range(7):
            c = base + s
            v_b00 = 0 * p + 2 * t
            v_b01 = 0 * p + 2 * t + 1
            v_b10 = 1 * p + 2 * t
            v_b11 = 1 * p + 2 * t + 1
            w_c00 = (2 * t) * n + 0
            w_c01 = (2 * t + 1) * n + 0
            w_c10 = (2 * t) * n + 1
            w_c11 = (2 * t + 1) * n + 1
            
            if s == 0:
                U[0][c] = 1
                U[3][c] = 1
                V[v_b00][c] = 1
                V[v_b11][c] = 1
                W[w_c00][c] = 1
                W[w_c11][c] = 1
            elif s == 1:
                U[2][c] = 1
                U[3][c] = 1
                V[v_b00][c] = 1
                W[w_c10][c] = 1
                W[w_c11][c] = -1
            elif s == 2:
                U[0][c] = 1
                V[v_b01][c] = 1
                V[v_b11][c] = -1
                W[w_c01][c] = 1
                W[w_c11][c] = 1
            elif s == 3:
                U[3][c] = 1
                V[v_b10][c] = 1
                V[v_b00][c] = -1
                W[w_c00][c] = 1
                W[w_c10][c] = 1
            elif s == 4:
                U[0][c] = 1
                U[1][c] = 1
                V[v_b11][c] = 1
                W[w_c00][c] = -1
                W[w_c01][c] = 1
            elif s == 5:
                U[2][c] = 1
                U[0][c] = -1
                V[v_b00][c] = 1
                V[v_b01][c] = 1
                W[w_c11][c] = 1
            elif s == 6:
                U[1][c] = 1
                U[3][c] = -1
                V[v_b10][c] = 1
                V[v_b11][c] = 1
                W[w_c00][c] = 1
    
    return {"U": U, "V": V, "W": W}
