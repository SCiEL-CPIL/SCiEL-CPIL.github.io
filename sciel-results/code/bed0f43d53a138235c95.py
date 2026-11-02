"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen algorithm for 2x2 block matrices, each block being 3x3
    # This gives rank = 7 * 27 = 189 (schoolbook for each 3x3 block multiplication)
    
    rank = 7 * 27
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen's 7 multiplication formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # Each M_k is a 3x3 matrix. For each M_k, we need 27 terms (schoolbook 3x3).
    # For each term, U has coefficients from the A-block linear combination,
    # V has coefficients from the B-block linear combination,
    # W has 1 at the appropriate C position.
    
    # Define Strassen steps: (A_blocks with signs, B_blocks with signs)
    # A blocks: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    
    steps = [
        # M1 = (A00+A11)*(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)*B00
        ({2: 1, 3: 1}, {0: 1}),
        # M3 = A00*(B01-B11)
        ({0: 1}, {1: 1, 3: -1}),
        # M4 = A11*(B10-B00)
        ({3: 1}, {2: 1, 0: -1}),
        # M5 = (A00+A01)*B11
        ({0: 1, 1: 1}, {3: 1}),
        # M6 = (A10-A00)*(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        # M7 = (A01-A11)*(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    # C combinations: which C blocks each M_k contributes to, with signs
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    c_combos = [
        {0: 1, 1: 0, 2: 0, 3: 1},   # M1 -> C00 (+1), C11 (+1)
        {0: 0, 1: 0, 2: 1, 3: -1},  # M2 -> C10 (+1), C11 (-1)
        {0: 0, 1: 1, 2: 0, 3: 1},   # M3 -> C01 (+1), C11 (+1)
        {0: -1, 1: 0, 2: 1, 3: 0},  # M4 -> C00 (-1), C10 (+1)
        {0: 0, 1: 1, 2: 0, 3: 0},   # M5 -> C01 (+1)
        {0: 1, 1: 0, 2: 0, 3: 0},   # M6 -> C00 (+1)
        {0: -1, 1: 0, 2: 0, 3: 1},  # M7 -> C00 (-1), C11 (+1)
    ]
    
    # For each Strassen step s, for each schoolbook term (i, j, k) in 3x3:
    # i = row index in A block (0,1,2)
    # j = column index in A block / row index in B block (0,1,2)  
    # k = column index in B block (0,1,2)
    
    for s in range(7):
        a_coeffs, b_coeffs = steps[s]
        c_coeffs = c_combos[s]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    col = s * 27 + i * 9 + j * 3 + k
                    
                    # U: for each A block in a_coeffs, set coefficient at position
                    # corresponding to A_block[i][j]
                    for a_block, sign in a_coeffs.items():
                        # A_block is 0,1,2,3 corresponding to:
                        # 0: rows 0-2, cols 0-2 (A00)
                        # 1: rows 0-2, cols 3-5 (A01)
                        # 2: rows 3-5, cols 0-2 (A10)
                        # 3: rows 3-5, cols 3-5 (A11)
                        row_block = a_block // 2  # 0 or 1
                        col_block = a_block % 2   # 0 or 1
                        global_row = row_block * 3 + i
                        global_col = col_block * 3 + j
                        U[global_row * m + global_col][col] += sign
                    
                    # V: for each B block in b_coeffs, set coefficient at position
                    # corresponding to B_block[j][k]
                    for b_block, sign in b_coeffs.items():
                        # B_block is 0,1,2,3 corresponding to:
                        # 0: rows 0-2, cols 0-2 (B00)
                        # 1: rows 0-2, cols 3-5 (B01)
                        # 2: rows 3-5, cols 0-2 (B10)
                        # 3: rows 3-5, cols 3-5 (B11)
                        row_block = b_block // 2  # 0 or 1
                        col_block = b_block % 2   # 0 or 1
                        global_row = row_block * 3 + j
                        global_col = col_block * 3 + k
                        V[global_row * p + global_col][col] += sign
                    
                    # W: for each C block in c_coeffs, set coefficient at position
                    # corresponding to C_block[i][k]
                    # W[k*n + i] where C[i][k] is the result
                    for c_block, sign in c_coeffs.items():
                        if sign == 0:
                            continue
                        # C_block is 0,1,2,3 corresponding to:
                        # 0: rows 0-2, cols 0-2 (C00)
                        # 1: rows 0-2, cols 3-5 (C01)
                        # 2: rows 3-5, cols 0-2 (C10)
                        # 3: rows 3-5, cols 3-5 (C11)
                        row_block = c_block // 2  # 0 or 1
                        col_block = c_block % 2   # 0 or 1
                        global_row = row_block * 3 + i
                        global_col = col_block * 3 + k
                        # W index: k*n + i, where k is col index in C, i is row index in C
                        W[global_col * n + global_row][col] += sign
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
