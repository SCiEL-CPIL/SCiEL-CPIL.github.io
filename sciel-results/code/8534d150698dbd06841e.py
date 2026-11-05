import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    factors_u = []
    factors_v = []
    factors_w = []
    
    def add_factor(u_idx, v_idx, w_idx, val=1):
        factors_u.append((u_idx, val))
        factors_v.append((v_idx, val))
        factors_w.append((w_idx, val))
        
    def add_strassen_222(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # a_rows, a_cols, b_rows, b_cols, c_rows, c_cols are ranges or lists
        # Strassen for 2x2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        
        b00 = (b_rows[0], b_cols[0])
        b01 = (b_rows[0], b_cols[1])
        b10 = (b_rows[1], b_cols[0])
        b11 = (b_rows[1], b_cols[1])
        
        c00 = (c_rows[0], c_cols[0])
        c01 = (c_rows[0], c_cols[1])
        c10 = (c_rows[1], c_cols[0])
        c11 = (c_rows[1], c_cols[1])
        
        # M1 = (A22) (B11 - B21)
        # U: A22 (1), V: B11 (1), B21 (-1), W: C21 (1)
        # Wait, Strassen formulas:
        # M1 = (A22) (B11 - B21)
        # M2 = (A11 + A22) B21
        # M3 = A11 (B12 - B22)
        # M4 = (A21 - A11) B11
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A12) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to add factors for each M term.
        # Each M term is a product of a linear combo of A and linear combo of B.
        # This expands into multiple factors.
        
        # Helper to add a product of linear combos
        def add_product(u_terms, v_terms, w_terms):
            # u_terms: list of (idx, coeff)
            # v_terms: list of (idx, coeff)
            # w_terms: list of (idx, coeff)
            # This adds rank-1 tensors for each combination?
            # No, a product of linear combos is a sum of rank-1 tensors.
            # (sum u_i) * (sum v_j) = sum u_i v_j
            # Each term u_i v_j contributes to W via w_terms.
            # So we add factors for each (u_i, v_j) pair, with W being the sum of w_terms?
            # No, W is the output.
            # The factor is a triple (u_idx, v_idx, w_idx).
            # If U is linear combo, V is linear combo, W is linear combo?
            # The tensor is sum_k U_k V_k W_k.
            # If we have (sum a_i A_i) (sum b_j B_j) -> sum c_k C_k
            # We can implement this by expanding the products.
            
            for u_idx, u_coeff in u_terms:
                for v_idx, v_coeff in v_terms:
                    val = u_coeff * v_coeff
                    if val == 0: continue
                    # This product contributes to the W terms
                    for w_idx, w_coeff in w_terms:
                        if w_coeff != 0:
                            add_factor(u_idx, v_idx, w_idx, val * w_coeff)
                            
        # M1 = A11 * (B00 - B10)
        # U: [(a11, 1)], V: [(b00, 1), (b10, -1)], W: [(c10, 1), (c11, -1)]? 
        # Wait, M1 contributes to C10 (C21 in 1-based) and C11 (C22 in 1-based)?
        # Let's map indices carefully.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1 = A11 (B00 - B10)
        # A11 is a11. B00 is b00, B10 is b10.
        # Contributes to C00 (coeff 1) and C11 (coeff 1).
        add_product([(a11, 1)], [(b00, 1), (b10, -1)], [(c00, 1), (c11, 1)])
        
        # M2 = (A00 + A11) B10
        # Contributes to C10 (coeff 1) and C11 (coeff -1).
        add_product([(a00, 1), (a11, 1)], [(b10, 1)], [(c10, 1), (c11, -1)])
        
        # M3 = A00 (B01 - B11)
        # Contributes to C01 (coeff 1) and C11 (coeff 1).
        add_product([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)])
        
        # M4 = (A10 - A00) B00
        # Contributes to C00 (coeff 1) and C10 (coeff 1).
        add_product([(a10, 1), (a00, -1)], [(b00, 1)], [(c00, 1), (c10, 1)])
        
        # M5 = (A00 + A01) B11
        # Contributes to C00 (coeff -1) and C01 (coeff 1).
        add_product([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])
        
        # M6 = (A10 - A01) (B00 + B01)
        # Contributes to C11 (coeff 1).
        add_product([(a10, 1), (a01, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])
        
        # M7 = (A01 - A11) (B10 + B11)
        # Contributes to C00 (coeff 1).
        add_product([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

    def add_221(a_rows, a_cols, b_row, b_col, c_row, c_col):
        # 2x2x1 multiplication
        # A is 2x2, B is 2x1, C is 2x1
        a00 = (a_rows[0], a_cols[0])
        a01 = (a_rows[0], a_cols[1])
        a10 = (a_rows[1], a_cols[0])
        a11 = (a_rows[1], a_cols[1])
        b0 = (b_row[0], b_col)
        b1 = (b_row[1], b_col)
        c0 = (c_row[0], c_col)
        c1 = (c_row[1], c_col)
        
        # c0 = a00*b0 + a01*b1
        # c1 = a10*b0 + a11*b1
        add_factor(a00, b0, c0, 1)
        add_factor(a01, b1, c0, 1)
        add_factor(a10, b0, c1, 1)
        add_factor(a11, b1, c1, 1)

    def add_121(a_row, a_cols, b_row, b_col, c_row, c_col):
        # 1x2x1 multiplication
        # A is 1x2, B is 2x1, C is 1x1
        a0 = (a_row, a_cols[0])
        a1 = (a_row, a_cols[1])
        b0 = (b_row[0], b_col)
        b1 = (b_row[1], b_col)
        c0 = (c_row, c_col)
        
        # c0 = a0*b0 + a1*b1
        add_factor(a0, b0, c0, 1)
        add_factor(a1, b1, c0, 1)

    # --- Block 1: j in {0, 1} ---
    # A_cols: 0, 1
    # B_rows: 0, 1
    a_cols_1 = [0, 1]
    b_rows_1 = [0, 1]
    
    # A_top: rows 0,1; A_bot: row 2
    a_top_rows = [0, 1]
    a_bot_row = 2
    
    # B_L: cols 0,1,2; B_R: cols 3,4
    b_l_cols = [0, 1, 2]
    b_r_cols = [3, 4]
    
    # C_rows: 0,1,2
    c_rows = [0, 1, 2]
    
    # Part 1: A_bot * B (1x2x5)
    # For each col k in 0..4:
    for k in range(5):
        add_121(a_bot_row, a_cols_1, b_rows_1, k, 2, k)
        
    # Part 2: A_top * B_R (2x2x2)
    # A_top: rows 0,1 cols 0,1
    # B_R: rows 0,1 cols 3,4
    # C_top_right: rows 0,1 cols 3,4
    add_strassen_222(a_top_rows, a_cols_1, b_rows_1, b_r_cols, a_top_rows, b_r_cols)
    
    # Part 3: A_top * B_L (2x2x3)
    # Split B_L into B_L1 (cols 0,1) and B_L2 (col 2)
    # A_top * B_L1 (2x2x2)
    add_strassen_222(a_top_rows, a_cols_1, b_rows_1, [0, 1], a_top_rows, [0, 1])
    
    # A_top * B_L2 (2x2x1)
    add_221(a_top_rows, a_cols_1, b_rows_1, 2, a_top_rows, 2)
    
    # --- Block 2: j in {2, 3} ---
    # A_cols: 2, 3
    # B_rows: 2, 3
    a_cols_2 = [2, 3]
    b_rows_2 = [2, 3]
    
    # Part 1: A_bot * B (1x2x5)
    for k in range(5):
        add_121(a_bot_row, a_cols_2, b_rows_2, k, 2, k)
        
    # Part 2: A_top * B_R (2x2x2)
    add_strassen_222(a_top_rows, a_cols_2, b_rows_2, b_r_cols, a_top_rows, b_r_cols)
    
    # Part 3: A_top * B_L (2x2x3)
    add_strassen_222(a_top_rows, a_cols_2, b_rows_2, [0, 1], a_top_rows, [0, 1])
    add_221(a_top_rows, a_cols_2, b_rows_2, 2, a_top_rows, 2)
    
    # Convert factors to matrices
    rank = len(factors_u)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, (u_idx, val) in enumerate(factors_u):
        i, j = u_idx
        U[i * m + j][idx] = val
        
    for idx, (v_idx, val) in enumerate(factors_v):
        j, k = v_idx
        V[j * p + k][idx] = val
        
    for idx, (w_idx, val) in enumerate(factors_w):
        k, i = w_idx
        W[k * n + i][idx] = val
        
    return {"U": U, "V": V, "W": W}
