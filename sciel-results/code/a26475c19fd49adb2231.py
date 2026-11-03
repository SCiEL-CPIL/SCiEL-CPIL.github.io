def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for U (A part)
    # Indices in A (2x2 block): 00, 01, 10, 11
    # M1: 00+11
    # M2: 10+11
    # M3: 00
    # M4: 11
    # M5: 00+01
    # M6: 10-00
    # M7: 01-11
    u_coeffs = [
        {0: 1, 3: 1},    # M1
        {2: 1, 3: 1},    # M2
        {0: 1},          # M3
        {3: 1},          # M4
        {0: 1, 1: 1},    # M5
        {2: 1, 0: -1},   # M6
        {1: 1, 3: -1}    # M7
    ]

    # Strassen coefficients for V (B part)
    # Indices in B (2x2 block): 00, 01, 10, 11
    # M1: 00+11
    # M2: 00
    # M3: 01-11
    # M4: 10-00
    # M5: 11
    # M6: 00+01
    # M7: 10+11
    v_coeffs = [
        {0: 1, 3: 1},    # M1
        {0: 1},          # M2
        {1: 1, 3: -1},   # M3
        {2: 1, 0: -1},   # M4
        {3: 1},          # M5
        {0: 1, 1: 1},    # M6
        {2: 1, 3: 1}     # M7
    ]

    # W contributions
    # C00: M1+M4-M5+M7
    # C01: M3+M5
    # C10: M2+M4
    # C11: M1-M2+M3+M6
    w_coeffs = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]

    # Block definitions
    # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # A is 2x4. B is 4x4. C is 2x4.
    # A split: cols 0-1 (L), 2-3 (R)
    # B split: rows 0-1 (T), 2-3 (B); cols 0-1 (L), 2-3 (R)
    
    blocks = [
        # Top-Left: A_L, B_TL -> C_L
        (0, 0, 0, 0, 0), 
        # Bottom-Left: A_R, B_BL -> C_L
        (0, 2, 2, 0, 0),
        # Top-Right: A_L, B_TR -> C_R
        (0, 0, 0, 2, 2),
        # Bottom-Right: A_R, B_BR -> C_R
        (0, 2, 2, 2, 2)
    ]

    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_c_off in blocks:
        for k in range(7):
            # Fill U
            for idx, val in u_coeffs[k].items():
                i = 0 # row in A block
                j = idx # col in A block
                # A index: (a_r_off + i) * m + (a_c_off + j)
                u_idx = (a_r_off + i) * m + (a_c_off + j)
                U[u_idx][col] += val
                
            # Fill V
            for idx, val in v_coeffs[k].items():
                # B index: (b_r_off + i) * p + (b_c_off + j)
                # idx is 0..3 corresponding to 00, 01, 10, 11
                # 0: 0,0; 1: 0,1; 2: 1,0; 3: 1,1
                i = idx // 2
                j = idx % 2
                v_idx = (b_r_off + i) * p + (b_c_off + j)
                V[v_idx][col] += val

            # Fill W
            # c_c_off is the column offset in C (0 or 2)
            # c_row is 0 or 1
            for c_row in range(2):
                w_val = w_coeffs[c_row][k]
                if w_val != 0:
                    # C index: (c_c_off + c_col) * n + c_row
                    # c_col is 0 or 1 within the 2x2 output block
                    # Wait, C is 2x4.
                    # c_c_off is 0 for left half, 2 for right half.
                    # The output block is 2x2.
                    # c_col in block is 0 or 1.
                    # So global col is c_c_off + c_col.
                    # w_idx = (global_col) * n + c_row
                    for c_col in range(2):
                         # The w_coeffs are for the 2x2 output Z.
                         # Z[0,0] -> C[c_c_off+0, 0]
                         # Z[0,1] -> C[c_c_off+1, 0]
                         # Z[1,0] -> C[c_c_off+0, 1]
                         # Z[1,1] -> C[c_c_off+1, 1]
                         # My w_coeffs list was:
                         # C00, C01, C10, C11
                         # C00 is row 0, col 0.
                         # C01 is row 0, col 1.
                         # C10 is row 1, col 0.
                         # C11 is row 1, col 1.
                         
                         # So w_coeffs[c_row][k] gives coefficient for Z[c_row, c_col]?
                         # No, w_coeffs is indexed by [output_row][m_index].
                         # Let's re-verify w_coeffs structure.
                         # w_coeffs[0] is for Z[0,0] and Z[0,1]?
                         # My previous definition:
                         # w_coeffs[0] = C00 coeffs
                         # w_coeffs[1] = C01 coeffs
                         # w_coeffs[2] = C10 coeffs
                         # w_coeffs[3] = C11 coeffs
                         
                         # So I need to iterate over the 4 output positions.
                         pass # Placeholder for logic fix

            col += 1

    # The logic above for W was messy. Let's redo W filling.
    # We have 4 blocks. Each block generates 7 M terms.
    # Each M term contributes to 4 output entries of the 2x2 block.
    
    # Let's re-initialize and do it cleanly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for a_r_off, a_c_off, b_r_off, b_c_off, c_c_off in blocks:
        for k in range(7):
            # U
            for idx, val in u_coeffs[k].items():
                i = 0
                j = idx
                u_idx = (a_r_off + i) * m + (a_c_off + j)
                U[u_idx][col] = val # Only one A row involved per coeff? 
                # Wait, u_coeffs are for 2x2 A.
                # idx 0: A00, 1: A01, 2: A10, 3: A11.
                # i = idx // 2, j = idx % 2.
                i = idx // 2
                j = idx % 2
                u_idx = (a_r_off + i) * m + (a_c_off + j)
                U[u_idx][col] += val

            # V
            for idx, val in v_coeffs[k].items():
                i = idx // 2
                j = idx % 2
                v_idx = (b_r_off + i) * p + (b_c_off + j)
                V[v_idx][col] += val

            # W
            # Output positions: (0,0), (0,1), (1,0), (1,1) relative to block
            # w_coeffs[pos][k]
            for pos in range(4):
                val = w_coeffs[pos][k]
                if val != 0:
                    # pos 0: (0,0), pos 1: (0,1), pos 2: (1,0), pos 3: (1,1)
                    out_r = pos // 2
                    out_c = pos % 2
                    # Global C indices:
                    # Row: out_r
                    # Col: c_c_off + out_c
                    w_idx = (c_c_off + out_c) * n + out_r
                    W[w_idx][col] += val

            col += 1

    return {"U": U, "V": V, "W": W}
