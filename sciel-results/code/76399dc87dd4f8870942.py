import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block-Strassen: 2x2 blocks of 3x3
    # Rank = 7 (Strassen) * 27 (Schoolbook 3x3) = 189
    
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients
    # Each entry is (list of (block_r, block_s, coeff) for X, list of (block_r, block_s, coeff) for Y)
    # And result coefficients: list of (block_r, block_s, coeff)
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        {
            'X': [(0, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1), (1, 1, 1)]
        },
        {
            'X': [(1, 0, 1), (1, 1, 1)],
            'Y': [(0, 0, 1)],
            'C': [(1, 0, 1), (1, 1, -1)]
        },
        {
            'X': [(0, 0, 1)],
            'Y': [(0, 1, 1), (1, 1, -1)],
            'C': [(0, 1, 1), (1, 1, 1)]
        },
        {
            'X': [(1, 1, 1)],
            'Y': [(1, 0, 1), (0, 0, -1)],
            'C': [(0, 0, 1), (1, 0, 1)]
        },
        {
            'X': [(0, 0, 1), (0, 1, 1)],
            'Y': [(1, 1, 1)],
            'C': [(0, 0, -1), (0, 1, 1)]
        },
        {
            'X': [(1, 0, 1), (0, 0, -1)],
            'Y': [(0, 0, 1), (0, 1, 1)],
            'C': [(1, 1, 1)]
        },
        {
            'X': [(0, 1, 1), (1, 1, -1)],
            'Y': [(1, 0, 1), (1, 1, 1)],
            'C': [(0, 0, 1)]
        }
    ]
    
    col = 0
    for term in strassen_terms:
        X_coeffs = term['X']
        Y_coeffs = term['Y']
        C_coeffs = term['C']
        
        # Iterate over schoolbook 3x3 multiplication
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # This term corresponds to X[a,b] * Y[b,c]
                    
                    # Fill U
                    for (br, bs, coeff) in X_coeffs:
                        # A index: row = 3*br + a, col = 3*bs + b
                        idx_a_row = 3 * br + a
                        idx_a_col = 3 * bs + b
                        u_idx = idx_a_row * 6 + idx_a_col
                        U[u_idx][col] += coeff
                        
                    # Fill V
                    for (br, bs, coeff) in Y_coeffs:
                        # B index: row = 3*br + b, col = 3*bs + c
                        idx_b_row = 3 * br + b
                        idx_b_col = 3 * bs + c
                        v_idx = idx_b_row * 6 + idx_b_col
                        V[v_idx][col] += coeff
                        
                    # Fill W
                    for (br, bs, coeff) in C_coeffs:
                        # C index: row = 3*br + a, col = 3*bs + c
                        # W index convention: k*n + i (col*6 + row)
                        idx_c_row = 3 * br + a
                        idx_c_col = 3 * bs + c
                        w_idx = idx_c_col * 6 + idx_c_row
                        W[w_idx][col] += coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
