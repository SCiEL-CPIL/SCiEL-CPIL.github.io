"""Matrix multiplication 2x4 x 4x8 tensor decomposition."""
DIMENSIONS = [2, 4, 8]

def construct():
    # Use rank-10 decomposition for M_{2,2,4} (from Strassen-like approach)
    # Actually, let me use the standard approach:
    # M_{2,4,8} can be decomposed using block structure
    # A is 2x4, B is 4x8, C is 2x8
    #
    # Split A into two 2x2 blocks: A_L, A_R
    # Split B into four 2x4 blocks: B_TL, B_TR, B_BL, B_BR
    # C = A_L * B_TL + A_R * B_BL  (for cols 0-3)
    #   + A_L * B_TR + A_R * B_BR  (for cols 4-7)
    #
    # Each 2x2 * 2x4 multiplication can be done with rank 10
    # (using 2 Strassen 2x2x2 multiplications, each rank 7, but we can do better)
    #
    # Actually, for 2x2 * 2x4:
    # C = A * B where A is 2x2, B is 2x4
    # C[:, 0:2] = A * B[:, 0:2] (one 2x2x2 product, rank 7)
    # C[:, 2:4] = A * B[:, 2:4] (another 2x2x2 product, rank 7)
    # Total: rank 14 per 2x2*2x4
    #
    # 4 such products: 4 * 14 = 56
    #
    # But we can potentially share some computations.
    # Let's use the straightforward rank-56 approach first, then try to optimize.
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen for 2x2 * 2x2:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b01-b11)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    strassen_products = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]
    
    c_reconstruction = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)],  # C11
    ]
    
    # 8 Strassen instances for the 8 2x2x2 sub-problems
    # (a_col_off, b_row_off, b_col_off)
    instances = [
        (0, 0, 0),  # A_L * B[0:2, 0:2]
        (0, 0, 2),  # A_L * B[0:2, 2:4]
        (2, 2, 0),  # A_R * B[2:4, 0:2]
        (2, 2, 2),  # A_R * B[2:4, 2:4]
        (0, 0, 4),  # A_L * B[0:2, 4:6]
        (0, 0, 6),  # A_L * B[0:2, 6:8]
        (2, 2, 4),  # A_R * B[2:4, 4:6]
        (2, 2, 6),  # A_R * B[2:4, 6:8]
    ]
    
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for prod_idx in range(7):
            a_factors, b_factors = strassen_products[prod_idx]
            for a_elem, a_coeff in a_factors:
                a_row = a_elem // 2
                a_col = a_elem % 2
                u_idx = a_row * m + (a_col_off + a_col)
                U[u_idx][col] += a_coeff
            for b_elem, b_coeff in b_factors:
                b_row = b_elem // 2
                b_col = b_elem % 2
                v_idx = (b_row_off + b_row) * p + (b_col_off + b_col)
                V[v_idx][col] += b_coeff
            col += 1
    
    col = 0
    for inst_idx, (a_col_off, b_row_off, b_col_off) in enumerate(instances):
        for c_elem in range(4):
            c_row = c_elem // 2
            c_col = c_elem % 2
            w_idx = (b_col_off + c_col) * n + c_row
            for prod_idx, coeff in c_reconstruction[c_elem]:
                W[w_idx][col + prod_idx] += coeff
        col += 7
    
    return {"U": U, "V": V, "W": W}
