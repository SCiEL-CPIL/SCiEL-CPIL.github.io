"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for M_{2,4,8}
    # Strategy: split A into two 2x2 blocks, split B into 4 2x4 blocks
    # Each 2x2 * 2x4 uses 2 Strassen 2x2x2 multiplications (rank 7 each)
    # Total: 8 Strassen instances * 7 = rank 56
    #
    # A (2x4): A_L = A[:, 0:2], A_R = A[:, 2:4]
    # B (4x8): B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #          B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    #
    # C = A_L * B_top + A_R * B_bottom
    # where B_top = B[0:2, :] (2x8) and B_bottom = B[2:4, :] (2x8)
    #
    # Each 2x2 * 2x4 product uses 2 Strassen instances:
    #   For B block (2x4) = [B_left (2x2) | B_right (2x2)]:
    #     A * B_left uses one Strassen
    #     A * B_right uses another Strassen

    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56

    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    # Strassen products for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    # Define Strassen product factors
    # Each product: (A_factors, B_factors) where factors are lists of (elem_idx, coeff)
    # A elem_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B elem_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # C reconstruction: for each C element, list of (prod_idx, coeff)
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]

    # 8 Strassen instances:
    # (a_col_off, b_row_off, b_col_off)
    # a_col_off: 0 for A_L, 2 for A_R
    # b_row_off: 0 for top 2 rows of B, 2 for bottom 2 rows
    # b_col_off: 0, 2, 4, 6 (global column offset in B)

    instances = [
        (0, 0, 0),  # A_L * B[0:2, 0:2]
        (0, 0, 2),  # A_L * B[0:2, 2:4]
        (0, 0, 4),  # A_L * B[0:2, 4:6]
        (0, 0, 6),  # A_L * B[0:2, 6:8]
        (2, 2, 0),  # A_R * B[2:4, 0:2]
        (2, 2, 2),  # A_R * B[2:4, 2:4]
        (2, 2, 4),  # A_R * B[2:4, 4:6]
        (2, 2, 6),  # A_R * B[2:4, 6:8]
    ]

    # Fill U and V
    col = 0
    for a_col_off, b_row_off, b_col_off in instances:
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]

            # Update U
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx][col] += a_coeff

            # Update V
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx][col] += b_coeff

            col += 1

    # Fill W
    col = 0
    for a_col_off, b_row_off, b_col_off in instances:
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            # W index: (b_col_off + c_col) * n + c_row
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx][col + prod_idx] += coeff
        col += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
