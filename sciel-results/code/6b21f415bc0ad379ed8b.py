import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for each of the 3 blocks
    for b in range(3):
        c_base = 7 * b
        # b11, b12, b21, b22 indices in V
        v_b11 = 0 * p + (2 * b)
        v_b12 = 0 * p + (2 * b + 1)
        v_b21 = 1 * p + (2 * b)
        v_b22 = 1 * p + (2 * b + 1)
        
        # a11, a12, a21, a22 indices in U
        u_a11 = 0 * m + 0
        u_a12 = 0 * m + 1
        u_a21 = 1 * m + 0
        u_a22 = 1 * m + 1
        
        # c11, c12, c21, c22 indices in W
        w_c11 = (2 * b) * n + 0
        w_c12 = (2 * b + 1) * n + 0
        w_c21 = (2 * b) * n + 1
        w_c22 = (2 * b + 1) * n + 1
        
        # M1
        c = c_base + 0
        U[u_a11][c] += 1
        U[u_a22][c] += 1
        V[v_b11][c] += 1
        V[v_b22][c] += 1
        W[w_c11][c] += 1
        W[w_c22][c] += 1
        
        # M2
        c = c_base + 1
        U[u_a21][c] += 1
        U[u_a22][c] += 1
        V[v_b11][c] += 1
        W[w_c21][c] += 1
        W[w_c22][c] += -1
        
        # M3
        c = c_base + 2
        U[u_a11][c] += 1
        V[v_b12][c] += 1
        V[v_b22][c] += -1
        W[w_c12][c] += 1
        W[w_c22][c] += 1
        
        # M4
        c = c_base + 3
        U[u_a22][c] += 1
        V[v_b21][c] += 1
        V[v_b11][c] += -1
        W[w_c11][c] += 1
        W[w_c21][c] += 1
        
        # M5
        c = c_base + 4
        U[u_a11][c] += 1
        U[u_a12][c] += 1
        V[v_b22][c] += 1
        W[w_c11][c] += -1
        W[w_c12][c] += 1
        
        # M6
        c = c_base + 5
        U[u_a21][c] += 1
        U[u_a11][c] += -1
        V[v_b11][c] += 1
        V[v_b12][c] += 1
        W[w_c22][c] += -1
        
        # M7
        c = c_base + 6
        U[u_a12][c] += 1
        U[u_a22][c] += -1
        V[v_b21][c] += 1
        V[v_b22][c] += 1
        W[w_c11][c] += 1
        
    return {"U": U, "V": V, "W": W}
