def construct():
    n, m, p = 3, 4, 7
    rank = 78
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_row_offset, a_col_offset: top-left of A block
    # b_row_offset, b_col_offset: top-left of B block
    # c_row_offset, c_col_offset: top-left of C block
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # Local U, V, W vectors for 7 mults
        # U: 4 elements (2x2 A)
        # V: 4 elements (2x2 B)
        # W: 4 elements (2x2 C)
        
        u_vecs = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        v_vecs = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        # W coefficients for C00, C01, C10, C11
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        w_coeffs = [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ]
        
        for k in range(7):
            # Add U
            for lr in range(2):
                for lc in range(2):
                    val = u_vecs[k][lr * 2 + lc]
                    if val != 0:
                        gr = a_r + lr
                        gc = a_c + lc
                        U[gr * m + gc][col] += val
            # Add V
            for lr in range(2):
                for lc in range(2):
                    val = v_vecs[k][lr * 2 + lc]
                    if val != 0:
                        gr = b_r + lr
                        gc = b_c + lc
                        V[gr * p + gc][col] += val
            # Add W
            for lr in range(2):
                for lc in range(2):
                    val = w_coeffs[lr * 2 + lc][k]
                    if val != 0:
                        gr = c_r + lr
                        gc = c_c + lc
                        W[gr * n + gc][col] += val
            col += 1

    # Helper to add a Schoolbook 2x2x1 block
    # A is 2x2, B is 2x1, C is 2x1
    def add_schoolbook_221(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        # Rank 4
        # Term 1: A00 * B00 -> C0 +=
        U[a_r * m + a_c][col] = 1
        V[b_r * p + b_c][col] = 1
        W[(c_r) * n + c_c][col] = 1
        col += 1
        
        # Term 2: A01 * B10 -> C0 +=
        U[a_r * m + a_c + 1][col] = 1
        V[(b_r + 1) * p + b_c][col] = 1
        W[(c_r) * n + c_c][col] = 1
        col += 1
        
        # Term 3: A10 * B00 -> C1 +=
        U[(a_r + 1) * m + a_c][col] = 1
        V[b_r * p + b_c][col] = 1
        W[(c_r + 1) * n + c_c][col] = 1
        col += 1
        
        # Term 4: A11 * B10 -> C1 +=
        U[(a_r + 1) * m + a_c + 1][col] = 1
        V[(b_r + 1) * p + b_c][col] = 1
        W[(c_r + 1) * n + c_c][col] = 1
        col += 1

    # Helper to add a Schoolbook 1x2x1 block (part of 1x4x4 or 1x4x3)
    # Actually, for 1x4x4 and 1x4x3, we can just iterate schoolbook.
    # Let's make a generic schoolbook adder for 1 x M x P
    def add_schoolbook_1xMp(a_r, a_c, b_r, b_c, c_r, c_c, m_block, p_block):
        nonlocal col
        for j in range(m_block):
            for k in range(p_block):
                U[a_r * m + a_c + j][col] = 1
                V[(b_r + j) * p + (b_c + k)][col] = 1
                W[(c_r + k) * n + c_c][col] = 1
                col += 1

    # Block 1: A_top (2x4) @ B_right (4x4) -> C_top_right (2x4)
    # A_top rows 0,1. C_top rows 0,1.
    # B_right cols 3,4,5,6.
    
    # Sub 1a: A_top_left (2x2) @ B_right_top (2x4) -> C_top_right_top (2x2)
    # A cols 0,1. B rows 0,1. C cols 3,4.
    # Split B_right_top (2x4) into (2x2) and (2x2).
    # B_right_top_left: rows 0,1 cols 3,4.
    # B_right_top_right: rows 0,1 cols 5,6.
    
    add_strassen(0, 0, 0, 3, 0, 3) # A(0,0), B(0,3), C(0,3) -> 2x2x2
    add_strassen(0, 0, 0, 5, 0, 5) # A(0,0), B(0,5), C(0,5) -> 2x2x2
    
    # Sub 1b: A_top_right (2x2) @ B_right_bot (2x4) -> C_top_right_bot (2x2)
    # A cols 2,3. B rows 2,3. C cols 3,4? No, C cols 5,6.
    # Wait, C = A_top B_right.
    # C_top_right is 2x4.
    # C_top_right = A_top_left B_right_top + A_top_right B_right_bot.
    # A_top_left is cols 0,1 of A_top.
    # A_top_right is cols 2,3 of A_top.
    # B_right_top is rows 0,1 of B_right.
    # B_right_bot is rows 2,3 of B_right.
    # B_right is cols 3,4,5,6 of B.
    # So B_right_top is B[0:2, 3:7].
    # B_right_bot is B[2:4, 3:7].
    
    # My previous split:
    # B_right_top (2x4) split into (2x2) and (2x2).
    # B_right_top_left: B[0:2, 3:5].
    # B_right_top_right: B[0:2, 5:7].
    
    # B_right_bot (2x4) split into (2x2) and (2x2).
    # B_right_bot_left: B[2:4, 3:5].
    # B_right_bot_right: B[2:4, 5:7].
    
    # So:
    # C_top_right_left (2x2) = A_top_left B_right_top_left + A_top_right B_right_bot_left
    # C_top_right_right (2x2) = A_top_left B_right_top_right + A_top_right B_right_bot_right
    
    # I already did:
    # add_strassen(0, 0, 0, 3, 0, 3) -> A(0,0) B(0,3) C(0,3). This is A_top_left, B_right_top_left, C_top_right_left.
    # add_strassen(0, 0, 0, 5, 0, 5) -> A(0,0) B(0,5) C(0,5). This is A_top_left, B_right_top_right, C_top_right_right.
    
    # Now I need:
    # A_top_right (cols 2,3) with B_right_bot_left (rows 2,3, cols 3,4) -> C_top_right_left (cols 3,4)
    add_strassen(0, 2, 2, 3, 0, 3)
    
    # A_top_right (cols 2,3) with B_right_bot_right (rows 2,3, cols 5,6) -> C_top_right_right (cols 5,6)
    add_strassen(0, 2, 2, 5, 0, 5)
    
    # Block 2: A_top (2x4) @ B_left (4x3) -> C_top_left (2x3)
    # B_left is B[0:4, 0:3].
    # B_left_top: B[0:2, 0:3].
    # B_left_bot: B[2:4, 0:3].
    
    # C_top_left_left (2x2) = A_top_left B_left_top_left + A_top_right B_left_bot_left
    # C_top_left_right (2x1) = A_top_left B_left_top_right + A_top_right B_left_bot_right
    
    # B_left_top (2x3) split into (2x2) and (2x1).
    # B_left_top_left: B[0:2, 0:2].
    # B_left_top_right: B[0:2, 2:3].
    
    # B_left_bot (2x3) split into (2x2) and (2x1).
    # B_left_bot_left: B[2:4, 0:2].
    # B_left_bot_right: B[2:4, 2:3].
    
    # 1. A_top_left (0,0) @ B_left_top_left (0,0) -> C_top_left_left (0,0)
    add_strassen(0, 0, 0, 0, 0, 0)
    
    # 2. A_top_left (0,0) @ B_left_top_right (0,2) -> C_top_left_right (0,2)
    # B_left_top_right is 2x1.
    add_schoolbook_221(0, 0, 0, 2, 0, 2)
    
    # 3. A_top_right (0,2) @ B_left_bot_left (2,0) -> C_top_left_left (0,0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # 4. A_top_right (0,2) @ B_left_bot_right (2,2) -> C_top_left_right (0,2)
    add_schoolbook_221(0, 2, 2, 2, 0, 2)
    
    # Block 3: A_bot (1x4) @ B_right (4x4) -> C_bot_right (1x4)
    # A_bot is row 2.
    # B_right is cols 3,4,5,6.
    add_schoolbook_1xMp(2, 0, 0, 3, 2, 3, 4, 4)
    
    # Block 4: A_bot (1x4) @ B_left (4x3) -> C_bot_left (1x3)
    add_schoolbook_1xMp(2, 0, 0, 0, 2, 0, 4, 3)
    
    return {"U": U, "V": V, "W": W}
