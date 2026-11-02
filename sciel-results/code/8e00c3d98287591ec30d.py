import numpy as np
import time

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using 3 Strassen blocks
    rank = 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2x2
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # Coefficients for A (row, col) -> (i, j)
    # Indices in U: i*m + j
    A_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},            # M3
        {(1,1): 1},            # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}  # M7
    ]
    
    # Coefficients for B (row, col) -> (j, l) where l is local col 0,1
    # Indices in V: j*p + k. Here k = 2*b + l
    B_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},            # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},            # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1}   # M7
    ]
    
    # Coefficients for C (row, col) -> (i, l) where l is local col 0,1
    # c00 = M1 - M2 + M3 - M7
    # c01 = M3 + M5
    # c10 = M4 + M6
    # c11 = M1 - M4 + M5 - M6
    C_coeffs = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): -1},           # M2
        {(0,0): 1, (0,1): 1},  # M3
        {(1,0): 1, (1,1): -1}, # M4
        {(0,1): 1, (1,1): 1},  # M5
        {(1,0): 1, (1,1): -1}, # M6
        {(0,0): -1}            # M7
    ]
    
    # 3 blocks
    for b in range(3):
        col_offset = b * 7
        for t in range(7):
            c = col_offset + t
            
            # Fill U
            for (i, j), val in A_coeffs[t].items():
                U[i * m + j][c] = val
                
            # Fill V
            for (j, l), val in B_coeffs[t].items():
                k = 2 * b + l
                V[j * p + k][c] = val
                
            # Fill W
            for (i, l), val in C_coeffs[t].items():
                k = 2 * b + l
                W[k * n + i][c] = val
                
    return {"U": U, "V": V, "W": W}
