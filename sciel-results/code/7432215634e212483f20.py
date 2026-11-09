# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen 2x2x2: rank 7
    strassen_products = [
        # P1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 32 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28

    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(a_row_offset, b_row_offset, b_col_offset, c_col_offset):
        nonlocal col
        for u_coeff, v_coeff, w_coeff in strassen_products:
            for (li, lj), coeff in u_coeff.items():
                gi = li
                gj = a_row_offset + lj
                U[gi * m + gj][col] = coeff
            for (lj, lk), coeff in v_coeff.items():
                gj = b_row_offset + lj
                gk = b_col_offset + lk
                V[gj * p + gk][col] = coeff
            for (lk, li), coeff in w_coeff.items():
                gk = b_col_offset + lk
                gi = li
                W[gk * n + gi][col] = coeff
            col += 1

    # Block 1: A_1 B_11 (A cols 0,1; B rows 0,1; B cols 0,1)
    add_strassen_block(0, 0, 0, 0)
    # Block 2: A_1 B_12 (A cols 0,1; B rows 0,1; B cols 2,3)
    add_strassen_block(0, 0, 2, 2)
    # Block 3: A_2 B_21 (A cols 2,3; B rows 2,3; B cols 0,1)
    add_strassen_block(2, 2, 0, 0)
    # Block 4: A_2 B_22 (A cols 2,3; B rows 2,3; B cols 2,3)
    add_strassen_block(2, 2, 2, 2)

    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
