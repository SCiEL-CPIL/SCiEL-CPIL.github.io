import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    
    # Rank 59 decomposition
    # 1. 8 Strassen blocks for 4x4x4 core (indices 0-3)
    # 2. 3 terms for the cross (indices involving 4)
    
    total_rank = 59
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks
    # I in {0,1} x {2,3}, J in {0,1} x {2,3}, K in {0,1} x {2,3}
    I_groups = [(0, 1), (2, 3)]
    J_groups = [(0, 1), (2, 3)]
    K_groups = [(0, 1), (2, 3)]
    
    def add_strassen(I, J, K):
        nonlocal col
        i1, i2 = I
        j1, j2 = J
        k1, k2 = K
        
        # M1 = (a11 + a22)(b11 + b22)
        U[i1 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[i2 * m + j1][col] += 1
        U[i2 * m + j2][col] += 1
        V[j1 * p + k1][col] += 1
        W[k1 * n + i2][col] += 1
        W[k2 * n + i2][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[i1 * m + j1][col] += 1
        V[j1 * p + k2][col] += 1
        V[j2 * p + k2][col] += -1
        W[k2 * n + i1][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[i2 * m + j2][col] += 1
        V[j2 * p + k1][col] += 1
        V[j1 * p + k1][col] += -1
        W[k1 * n + i1][col] += 1
        W[k1 * n + i2][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[i1 * m + j1][col] += 1
        U[i1 * m + j2][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += -1
        W[k2 * n + i1][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[i2 * m + j1][col] += 1
        U[i1 * m + j1][col] += -1
        V[j1 * p + k1][col] += 1
        V[j1 * p + k2][col] += 1
        W[k2 * n + i2][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[i1 * m + j2][col] += 1
        U[i2 * m + j2][col] += -1
        V[j2 * p + k1][col] += 1
        V[j2 * p + k2][col] += 1
        W[k1 * n + i1][col] += 1
        col += 1

    for I in I_groups:
        for J in J_groups:
            for K in K_groups:
                add_strassen(I, J, K)
                
    # Cross terms
    # R1: Covers T[i, 4, k] for all i, k in 0..3
    # U[i, 4] = 1, V[4, k] = 1, W[k, i] = 1
    for i in range(4):
        U[i * m + 4][col] = 1
    for k in range(4):
        V[4 * p + k][col] = 1
    for k in range(4):
        for i in range(4):
            W[k * n + i][col] = 1
    col += 1
    
    # R2: Covers T[i, j, 4] for all i, j in 0..3
    # U[i, j] = 1, V[j, 4] = 1, W[4, i] = 1
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
    for j in range(4):
        V[j * p + 4][col] = 1
    for i in range(4):
        W[4 * n + i][col] = 1
    col += 1
    
    # R3: Correction for T[i, 4, 4]
    # U[i, 4] = -1, V[4, 4] = 1, W[4, i] = 1
    for i in range(4):
        U[i * m + 4][col] = -1
    V[4 * p + 4][col] = 1
    for i in range(4):
        W[4 * n + i][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
