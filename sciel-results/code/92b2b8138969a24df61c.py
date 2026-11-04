import numpy as np

def construct():
    n, m, p = 2, 4, 7
    R = 50
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    col = 0
    
    def add_strassen_block(a_row_offsets, a_col_offsets, b_row_offsets, b_col_offsets, c_row_offsets, c_col_offsets):
        nonlocal col
        # a_row_offsets: list of 2 ints (i indices in A)
        # a_col_offsets: list of 2 ints (j indices in A)
        # b_row_offsets: list of 2 ints (j indices in B)
        # b_col_offsets: list of 2 ints (k indices in B)
        # c_row_offsets: list of 2 ints (k indices in C) -> Wait, C is 2x7. 
        # W index is k*n + i. So c_row_offsets are k indices.
        # c_col_offsets: list of 2 ints (i indices in C)
        
        # Strassen M1..M7
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11.
        # We need to represent the product of sums.
        # This is not a single rank 1 term. 
        # Strassen is a decomposition into 7 rank 1 terms.
        # Each term is (linear combo of A) * (linear combo of B) * (contribution to C).
        
        # Term 1: (A00+A11) * (B00+B11)
        # U entries: 1 at (A00), 1 at (A11).
        # V entries: 1 at (B00), 1 at (B11).
        # W entries: 1 at (C00), -1 at (C01)? No.
        # C00 += M1. C01 += 0. C10 += 0. C11 -= M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M1 contributes +1 to C00, +1 to C11.
        
        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            for i_a, j_a, c_val in u_coeffs:
                U[i_a * m + j_a, col] += c_val
            for i_b, j_b, c_val in v_coeffs:
                V[i_b * p + j_b, col] += c_val
            for i_c, j_c, c_val in w_coeffs:
                W[i_c * n + j_c, col] += c_val
            col += 1

        # Helper to get global indices
        def u_idx(i, j): return i * m + j
        def v_idx(i, j): return i * p + j
        def w_idx(i, j): return i * n + j
        
        # M1
        add_term(
            [(a_row_offsets[0], a_col_offsets[0], 1), (a_row_offsets[1], a_col_offsets[1], 1)],
            [(b_row_offsets[0], b_col_offsets[0], 1), (b_row_offsets[1], b_col_offsets[1], 1)],
            [(c_row_offsets[0], c_col_offsets[0], 1), (c_row_offsets[1], c_col_offsets[1], 1)]
        )
        
        # M2: (A10+A11) * B00
        add_term(
            [(a_row_offsets[1], a_col_offsets[0], 1), (a_row_offsets[1], a_col_offsets[1], 1)],
            [(b_row_offsets[0], b_col_offsets[0], 1)],
            [(c_row_offsets[1], c_col_offsets[0], 1), (c_row_offsets[1], c_col_offsets[1], -1)]
        )
        
        # M3: A00 * (B01-B11)
        add_term(
            [(a_row_offsets[0], a_col_offsets[0], 1)],
            [(b_row_offsets[0], b_col_offsets[1], 1), (b_row_offsets[1], b_col_offsets[1], -1)],
            [(c_row_offsets[0], c_col_offsets[1], 1), (c_row_offsets[1], c_col_offsets[1], 1)]
        )
        
        # M4: A11 * (B10-B00)
        add_term(
            [(a_row_offsets[1], a_col_offsets[1], 1)],
            [(b_row_offsets[1], b_col_offsets[0], 1), (b_row_offsets[0], b_col_offsets[0], -1)],
            [(c_row_offsets[0], c_col_offsets[0], 1), (c_row_offsets[1], c_col_offsets[0], 1)]
        )
        
        # M5: (A00+A01) * B11
        add_term(
            [(a_row_offsets[0], a_col_offsets[0], 1), (a_row_offsets[0], a_col_offsets[1], 1)],
            [(b_row_offsets[1], b_col_offsets[1], 1)],
            [(c_row_offsets[0], c_col_offsets[0], -1), (c_row_offsets[0], c_col_offsets[1], 1)]
        )
        
        # M6: (A10-A00) * (B00+B01)
        add_term(
            [(a_row_offsets[1], a_col_offsets[0], 1), (a_row_offsets[0], a_col_offsets[0], -1)],
            [(b_row_offsets[0], b_col_offsets[0], 1), (b_row_offsets[0], b_col_offsets[1], 1)],
            [(c_row_offsets[1], c_col_offsets[1], 1)]
        )
        
        # M7: (A01-A11) * (B10+B11)
        add_term(
            [(a_row_offsets[0], a_col_offsets[1], 1), (a_row_offsets[1], a_col_offsets[1], -1)],
            [(b_row_offsets[1], b_col_offsets[0], 1), (b_row_offsets[1], b_col_offsets[1], 1)],
            [(c_row_offsets[0], c_col_offsets[0], 1)]
        )

    # Block 1: M_2,4,4
    # A1 (cols 0-1) * B_top (rows 0-1, cols 0-1) -> C (rows 0-1, cols 0-1)
    # Note: C indices in W are k*n + i. 
    # C is 2x7. i is row of C (0-1), k is col of C (0-6).
    # W index = k * 2 + i.
    # So c_row_offsets are k indices. c_col_offsets are i indices.
    
    # A1: rows 0-1, cols 0-1.
    # B_top_01: rows 0-1, cols 0-1.
    # C_01: cols 0-1, rows 0-1.
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # A1: rows 0-1, cols 0-1.
    # B_top_23: rows 0-1, cols 2-3.
    # C_23: cols 2-3, rows 0-1.
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [2, 3], [0, 1])
    
    # A2: rows 0-1, cols 2-3.
    # B_bot_01: rows 2-3, cols 0-1.
    # C_01: cols 0-1, rows 0-1.
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # A2: rows 0-1, cols 2-3.
    # B_bot_23: rows 2-3, cols 2-3.
    # C_23: cols 2-3, rows 0-1.
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [2, 3], [0, 1])
    
    # Block 2: M_2,4,2 (Cols 4-5)
    # A1 * B_top_45 -> C_45
    add_strassen_block([0, 1], [0, 1], [0, 1], [4, 5], [4, 5], [0, 1])
    
    # A2 * B_bot_45 -> C_45
    add_strassen_block([0, 1], [2, 3], [2, 3], [4, 5], [4, 5], [0, 1])
    
    # Block 3: M_2,4,1 (Col 6)
    # Schoolbook
    for i in range(2):
        for j in range(4):
            U[i * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + i, col] = 1
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
