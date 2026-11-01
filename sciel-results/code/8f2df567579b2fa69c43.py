"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 7 blocks of 2x2 (Strassen rank 7) + 1 block of 2x1 (Schoolbook rank 4)
    # Total rank = 7 * 7 + 4 = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2 matrix multiplication
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # U vectors (A coefficients)
    # A indices: 0:00, 1:01, 2:10, 3:11
    U_strassen = [
        [1, 0, 0, 1], # m1: a+d
        [0, 0, 1, 1], # m2: c+d
        [1, 0, 0, 0], # m3: a
        [0, 0, 0, 1], # m4: d
        [1, 1, 0, 0], # m5: a+b
        [-1, 0, 1, 0],# m6: c-a
        [0, 1, 0, -1] # m7: b-d
    ]
    
    # V vectors (B coefficients) for a 2x2 block
    # B indices in block: 0:00, 1:01, 2:10, 3:11
    V_strassen = [
        [1, 0, 0, 1], # m1: e+h
        [1, 0, 0, 0], # m2: e
        [0, 1, 0, -1],# m3: f-h
        [0, 0, 1, -1],# m4: g-e
        [0, 0, 0, 1], # m5: h
        [1, 1, 0, 0], # m6: e+f
        [0, 0, 1, 1]  # m7: g+h
    ]
    
    # W vectors (C coefficients)
    # C indices in block: 0:00, 1:01, 2:10, 3:11
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    W_strassen = [
        [1, 0, 0, 1], # m1
        [0, 0, 1, -1],# m2
        [0, 1, 0, 1], # m3
        [1, 0, 1, 0], # m4
        [-1, 1, 0, 0],# m5
        [0, 0, 0, 1], # m6
        [1, 0, 0, 0]  # m7
    ]
    
    # Process 7 blocks of size 2x2
    for t in range(7):
        for s in range(7):
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_strassen[s][i * 2 + j]
            
            # Fill V
            # B block t starts at column 2*t
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * t + k_local
                    V[j * p + k_global][col] = V_strassen[s][j * 2 + k_local]
            
            # Fill W
            # C block t starts at column 2*t
            for i in range(2):
                for k_local in range(2):
                    k_global = 2 * t + k_local
                    W[k_global * n + i][col] = W_strassen[s][k_local * 2 + i]
            
            col += 1
    
    # Process last block (column 14) with schoolbook
    # C0,14 = A0,0 B0,14 + A0,1 B1,14
    # C1,14 = A1,0 B0,14 + A1,1 B1,14
    
    # Term 1: A0,0 * B0,14 -> C0,14
    U[0][col] = 1
    V[0 * p + 14][col] = 1
    W[14 * n + 0][col] = 1
    col += 1
    
    # Term 2: A0,1 * B1,14 -> C0,14
    U[1][col] = 1
    V[1 * p + 14][col] = 1
    W[14 * n + 0][col] = 1
    col += 1
    
    # Term 3: A1,0 * B0,14 -> C1,14
    U[2][col] = 1
    V[0 * p + 14][col] = 1
    W[14 * n + 1][col] = 1
    col += 1
    
    # Term 4: A1,1 * B1,14 -> C1,14
    U[3][col] = 1
    V[1 * p + 14][col] = 1
    W[14 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
