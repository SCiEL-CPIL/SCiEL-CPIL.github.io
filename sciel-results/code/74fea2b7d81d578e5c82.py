DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 matrix over F^5
    # A_coeffs[s]: list of (A_index, coeff)
    # B_coeffs[s]: list of (B_row, B_block, coeff) where B_block 0->cols 0-4, 1->cols 5-9
    # W_coeffs[s]: (C00, C01, C10, C11) coefficients
    
    A_coeffs = [
        [(0, 1), (3, 1)],    # M1
        [(2, 1), (3, 1)],    # M2
        [(0, 1)],            # M3
        [(3, 1)],            # M4
        [(0, 1), (1, 1)],    # M5
        [(2, 1), (0, -1)],   # M6
        [(1, 1), (3, -1)]    # M7
    ]
    
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1
        [(0, 0, 1)],               # M2
        [(0, 1, 1), (1, 1, -1)],  # M3
        [(1, 0, 1), (0, 0, -1)],  # M4
        [(1, 1, 1)],               # M5
        [(0, 0, 1), (0, 1, 1)],   # M6
        [(1, 0, 1), (1, 1, 1)]    # M7
    ]
    
    W_coeffs = [
        (1, 0, 0, 1),    # M1 -> C00, C11
        (0, 0, 1, -1),   # M2 -> C10, C11
        (0, 1, 0, 1),    # M3 -> C01, C11
        (1, 0, 1, 0),    # M4 -> C00, C10
        (-1, 1, 0, 0),   # M5 -> C00, C01
        (0, 0, 0, 1),    # M6 -> C11
        (1, 0, 0, 0)     # M7 -> C00
    ]
    
    col = 0
    for s in range(7):
        for t in range(5):
            # U
            for a_idx, a_c in A_coeffs[s]:
                U[a_idx][col] = a_c
            
            # V
            for b_row, b_block, b_c in B_coeffs[s]:
                b_col = b_block * 5 + t
                V[b_row * p + b_col][col] = b_c
            
            # W
            c00, c01, c10, c11 = W_coeffs[s]
            if c00 != 0: W[t * n + 0][col] = c00
            if c01 != 0: W[(5 + t) * n + 0][col] = c01
            if c10 != 0: W[t * n + 1][col] = c10
            if c11 != 0: W[(5 + t) * n + 1][col] = c11
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
