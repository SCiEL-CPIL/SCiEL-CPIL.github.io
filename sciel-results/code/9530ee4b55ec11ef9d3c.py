def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen products definitions
    # Each product is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {A_idx: coeff}
    # v_coeffs: dict {B_idx: coeff}
    # w_coeffs: dict {C_idx: coeff}
    # A_idx: 0=A00, 1=A01, 2=A10, 3=A11
    # B_idx: 0=B00, 1=B01, 2=B10, 3=B11
    # C_idx: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen_products = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1})
    ]
    
    # Define blocks
    # Block 0: A_L B_L0
    # A_L: A[:, 0:2] -> rows 0,1,4,5 in U
    # B_L0: B[0:2, 0:2] -> rows 0,1,4,5 in V
    # C_L0: C[:, 0:2] -> rows 0,2,1,3 in W
    blocks = [
        # (A_rows, B_rows, C_rows)
        # A_rows: [A00, A01, A10, A11]
        # B_rows: [B00, B01, B10, B11]
        # C_rows: [C00, C01, C10, C11]
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3]), # Block 0
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7]), # Block 1
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3]), # Block 2
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7]) # Block 3
    ]
    
    for b_idx, (a_rows, b_rows, c_rows) in enumerate(blocks):
        for m_idx, (u_c, v_c, w_c) in enumerate(strassen_products):
            col = b_idx * 7 + m_idx
            
            for a_idx, coeff in u_c.items():
                U[a_rows[a_idx]][col] += coeff
            
            for b_idx_val, coeff in v_c.items():
                V[b_rows[b_idx_val]][col] += coeff
            
            for c_idx, coeff in w_c.items():
                W[c_rows[c_idx]][col] += coeff
                
    return {"U": U, "V": V, "W": W}
