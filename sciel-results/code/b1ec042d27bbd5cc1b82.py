import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (16, rank)
    # W shape: (p*n, rank) = (8, rank)
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a tuple of (A_coeffs, B_coeffs, C_coeffs)
    # Coeffs are dicts mapping local (row, col) to value
    # Local indices are (0,0), (0,1), (1,0), (1,1)
    
    strassen_terms = [
        # P1
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        # P2
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        # P3
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        # P4
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        # P5
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        # P6
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        # P7
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
    ]
    
    # Block definitions
    # (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4, rows 0,1 are always used.
    # B is 4x4.
    # C is 2x4, rows 0,1 are always used.
    
    blocks = [
        # M1 = A_L * B_TL -> C_L
        (0, 0, 0, 0),
        # M2 = A_R * B_BL -> C_L
        (2, 2, 0, 0),
        # M3 = A_L * B_TR -> C_R
        (0, 0, 2, 2),
        # M4 = A_R * B_BR -> C_R
        (2, 2, 2, 2),
    ]
    
    col_idx = 0
    for a_c, b_r, b_c, c_c in blocks:
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            # Process A coefficients
            for (lr, lc), val in a_coeffs.items():
                # Global A index: row lr, col a_c + lc
                # A is 2x4. Index = lr * 4 + (a_c + lc)
                a_idx = lr * m + (a_c + lc)
                U[a_idx][col_idx] += val
            
            # Process B coefficients
            for (lr, lc), val in b_coeffs.items():
                # Global B index: row b_r + lr, col b_c + lc
                # B is 4x4. Index = (b_r + lr) * 4 + (b_c + lc)
                b_idx = (b_r + lr) * p + (b_c + lc)
                V[b_idx][col_idx] += val
            
            # Process C coefficients
            for (lr, lc), val in c_coeffs.items():
                # Global C index: row lr, col c_c + lc
                # W index convention: k * n + i
                # k is col of C, i is row of C
                # So k = c_c + lc, i = lr
                w_idx = (c_c + lc) * n + lr
                W[w_idx][col_idx] += val
            
            col_idx += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
