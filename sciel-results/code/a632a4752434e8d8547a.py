"""Matrix multiplication 2,4,6 - rank 42 decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    
    # Rank is 42: 2 blocks of A (2x2) x 3 blocks of B cols (2x2) x 7 Strassen terms
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # For A (n x m block, 2x2): U_coeffs[s] maps (i_local, j_local) -> coeff
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: (A00 + A11)
        {(1, 0): 1, (1, 1): 1},       # P2: (A10 + A11)
        {(0, 0): 1},                   # P3: A00
        {(1, 1): 1},                   # P4: A11
        {(0, 0): 1, (0, 1): 1},       # P5: (A00 + A01)
        {(1, 0): 1, (0, 0): -1},      # P6: (A10 - A00)
        {(0, 1): 1, (1, 1): -1}       # P7: (A01 - A11)
    ]
    
    # For B (m x p block, 2x2): V_coeffs[s] maps (j_local, k_local) -> coeff
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: (B00 + B11)
        {(0, 0): 1},                   # P2: B00
        {(0, 1): 1, (1, 1): -1},      # P3: (B01 - B11)
        {(1, 0): 1, (0, 0): -1},      # P4: (B10 - B00)
        {(1, 1): 1},                   # P5: B11
        {(0, 0): 1, (0, 1): 1},       # P6: (B00 + B01)
        {(1, 0): 1, (1, 1): 1}        # P7: (B10 + B11)
    ]
    
    # For C (n x p block, 2x2): W_coeffs[s] maps (i_local, k_local) -> coeff
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},       # P1: C00 += 1, C11 += 1
        {(1, 0): 1, (1, 1): -1},      # P2: C10 += 1, C11 += -1
        {(0, 1): 1, (1, 1): 1},       # P3: C01 += 1, C11 += 1
        {(0, 0): 1, (1, 0): 1},       # P4: C00 += 1, C10 += 1
        {(0, 0): -1, (0, 1): 1},      # P5: C00 += -1, C01 += 1
        {(1, 1): 1},                   # P6: C11 += 1
        {(0, 0): 1}                    # P7: C00 += 1
    ]
    
    col = 0
    for a in range(2):  # A column block: j in {0,1} or {2,3}
        for b in range(3):  # B column block: k in {0,1}, {2,3}, or {4,5}
            for s in range(7):  # Strassen term
                # Update U
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j][col] += coeff
                
                # Update V
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k][col] += coeff
                
                # Update W
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
