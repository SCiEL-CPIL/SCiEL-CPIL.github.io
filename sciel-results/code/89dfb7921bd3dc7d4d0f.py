import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    # Rank 28 Strassen-based decomposition
    # C (2x4) = A (2x4) * B (4x4)
    # Partition: C[:, 0:2] = A[:, 0:2]*B[0:2, 0:2] + A[:, 2:4]*B[2:4, 0:2]
    #             C[:, 2:4] = A[:, 0:2]*B[0:2, 2:4] + A[:, 2:4]*B[2:4, 2:4]
    # Each 2x2*2x2 product uses Strassen (rank 7), total rank = 4*7 = 28
    
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0.0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen coefficients for 2x2x2
    # U coeffs (A part, 4 elements per 2x2 block):
    U_strassen = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 1, -1, 0],  # M6: A01-A10
        [-1, 0, 0, 1],  # M7: A11-A00
    ]
    
    # V coeffs (B part, 4 elements per 2x2 block):
    V_strassen = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 0, 1, 0],   # M6: B00+B10
        [0, 1, 0, 1],   # M7: B11+B01
    ]
    
    # W coeffs (C part): each output element is a linear combo of M1..M7
    # C00 = M1 - M4 + M6 -> coeff [1, 0, 0, -1, 0, 1, 0]
    # C01 = M3 + M5      -> coeff [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4      -> coeff [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M3 + M7 -> coeff [1, 0, -1, 0, 0, 0, 1]
    W_strassen = [
        [1, 0, 0, -1, 0, 1, 0],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, 0, -1, 0, 0, 0, 1],   # C11
    ]
    
    # Four blocks:
    # Block 0: A[:, 0:2] * B[0:2, 0:2] -> contributes to C[:, 0:2]
    # Block 1: A[:, 2:4] * B[2:4, 0:2] -> contributes to C[:, 0:2]
    # Block 2: A[:, 0:2] * B[0:2, 2:4] -> contributes to C[:, 2:4]
    # Block 3: A[:, 2:4] * B[2:4, 2:4] -> contributes to C[:, 2:4]
    
    # For each block, A is 2x2 (rows 0,1 of A for the column range)
    # A indices: row i, col j -> index i*4 + j in U
    # B indices: row j, col k -> index j*4 + k in V
    # C indices: col k, row i -> index k*2 + i in W (trace convention W[k*n+i, r])
    
    # Block definitions: (a_col_start, b_row_start, b_col_start, c_col_start)
    blocks = [
        (0, 0, 0, 0),  # A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
        (2, 2, 0, 0),  # A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
        (0, 0, 2, 2),  # A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
        (2, 2, 2, 2),  # A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    ]
    
    for b_idx, (a_col_start, b_row_start, b_col_start, c_col_start) in enumerate(blocks):
        base = b_idx * 7
        for s in range(7):
            col = base + s
            # U: A is 2x2, elements at (row i, col a_col_start + j)
            # A element (i, j) in the 2x2 block -> U index i*4 + (a_col_start + j)
            # U_strassen[s] gives coeffs for [A00, A01, A10, A11]
            for i in range(2):
                for j in range(2):
                    coeff = U_strassen[s][i * 2 + j]
                    if coeff != 0:
                        U[i * 4 + (a_col_start + j)][col] += coeff
            
            # V: B is 2x2, elements at (row b_row_start + i, col b_col_start + j)
            # B element (i, j) in the 2x2 block -> V index (b_row_start + i)*4 + (b_col_start + j)
            # V_strassen[s] gives coeffs for [B00, B01, B10, B11]
            for i in range(2):
                for j in range(2):
                    coeff = V_strassen[s][i * 2 + j]
                    if coeff != 0:
                        V[(b_row_start + i) * 4 + (b_col_start + j)][col] += coeff
            
            # W: C output is 2x2, elements at (col c_col_start + j, row i)
            # C element (row i, col j) in the 2x2 block -> W index (c_col_start + j)*2 + i
            # W_strassen[c_out] gives coeffs for [M1, M2, M3, M4, M5, M6, M7]
            for c_out in range(4):
                coeff = W_strassen[c_out][s]
                if coeff != 0:
                    c_row = c_out // 2
                    c_col = c_out % 2
                    W[(c_col_start + c_col) * 2 + c_row][col] += coeff
    
    return {"U": U, "V": V, "W": W}
