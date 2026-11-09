"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: partition 4x4 matrix into four 2x2 blocks.
    # A is 2x4 -> partition into A_0 (cols 0-1) and A_1 (cols 2-3), each 2x2.
    # B is 4x4 -> partition into B_00 (rows 0-1, cols 0-1), B_01 (rows 0-1, cols 2-3),
    #            B_10 (rows 2-3, cols 0-1), B_11 (rows 2-3, cols 2-3), each 2x2.
    # C is 2x4 -> partition into C_0 (cols 0-1) and C_1 (cols 2-3), each 2x2.
    #
    # C_0 = A_0 * B_00 + A_1 * B_10
    # C_1 = A_0 * B_01 + A_1 * B_11
    #
    # Apply Strassen to each 2x2x2 multiplication:
    # M1 = (A_0 + A_1)(B_00 + B_11)  -- but we need separate products.
    # Actually, we have two independent 2x2x2 problems:
    #   Problem 1: C_0 = A_0 * B_00 + A_1 * B_10  (sum of two 2x2x2)
    #   Problem 2: C_1 = A_0 * B_01 + A_1 * B_11  (sum of two 2x2x2)
    #
    # Each 2x2x2 multiplication can be done with Strassen rank 7.
    # So total rank = 4 * 7 = 28.
    #
    # Strassen for 2x2:
    #   m1 = (a11 + a22)(b11 + b22)
    #   m2 = (a21 + a22) b11
    #   m3 = a11 (b12 - b22)
    #   m4 = a22 (b21 - b11)
    #   m5 = (a11 + a12) b22
    #   m6 = (a21 - a11)(b11 + b12)
    #   m7 = (a12 - a22)(b21 + b22)
    #   c11 = m1 + m4 - m5 + m7
    #   c12 = m3 + m5
    #   c21 = m2 + m4
    #   c22 = m1 - m2 + m3 + m6
    #
    # For each of the 4 sub-problems, we map local Strassen entries to global U, V, W.

    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen coefficients for the 7 products:
    # Each product has a linear combination of A entries and B entries.
    # We represent each of the 7 Strassen products as a rank-1 tensor.

    # Strassen product definitions: (A_coeff, B_coeff) where each is a list of (local_idx, weight)
    # Local indices for 2x2: 0=a11, 1=a12, 2=a21, 3=a22
    #                          0=b11, 1=b12, 2=b21, 3=b22

    strassen_products = [
        # m1 = (a11+a22)(b11+b22)
        ([((0, 1), (3, 1))], [(0, 1), (3, 1)]),
        # m2 = (a21+a22) b11
        ([((2, 1), (3, 1))], [(0, 1)]),
        # m3 = a11 (b12-b22)
        ([((0, 1),)], [(1, 1), (3, -1)]),
        # m4 = a22 (b21-b11)
        ([((3, 1),)], [(2, 1), (0, -1)]),
        # m5 = (a11+a12) b22
        ([((0, 1), (1, 1))], [(3, 1)]),
        # m6 = (a21-a11)(b11+b12)
        ([((2, 1), (0, -1))], [(0, 1), (1, 1)]),
        # m7 = (a12-a22)(b21+b22)
        ([((1, 1), (3, -1))], [(2, 1), (3, 1)]),
    ]

    # For each of the 4 sub-problems, define:
    #   - Which A block (A_0 or A_1)
    #   - Which B block (B_00, B_01, B_10, B_11)
    #   - Which C block (C_0 or C_1)
    #   - The Strassen combination weights for c11, c12, c21, c22

    # Sub-problem definitions: (a_block, b_block, c_block, c_weights)
    # c_weights: for each of the 4 output positions (c11, c12, c21, c22),
    #            a dict mapping strassen_product_index -> weight
    sub_problems = [
        # C_0 = A_0 * B_00 + A_1 * B_10
        # First: A_0 * B_00 -> contributes to C_0
        (0, 0, 0, {0: 1, 3: 1, 4: -1, 6: 1,  # c11 = m1+m4-m5+m7
                    2: 1, 4: 1,               # c12 = m3+m5
                    1: 1, 3: 1,               # c21 = m2+m4
                    0: 1, 1: -1, 2: 1, 5: 1}),  # c22 = m1-m2+m3+m6
        # Second: A_1 * B_10 -> contributes to C_0
        (1, 2, 0, {0: 1, 3: 1, 4: -1, 6: 1,
                    2: 1, 4: 1,
                    1: 1, 3: 1,
                    0: 1, 1: -1, 2: 1, 5: 1}),
        # C_1 = A_0 * B_01 + A_1 * B_11
        (0, 1, 1, {0: 1, 3: 1, 4: -1, 6: 1,
                    2: 1, 4: 1,
                    1: 1, 3: 1,
                    0: 1, 1: -1, 2: 1, 5: 1}),
        (1, 3, 1, {0: 1, 3: 1, 4: -1, 6: 1,
                    2: 1, 4: 1,
                    1: 1, 3: 1,
                    0: 1, 1: -1, 2: 1, 5: 1}),
    ]

    # Fix: the c_weights dict has duplicate keys. Let me restructure properly.
    # For each sub-problem, we need 4 output entries, each is a linear combination of the 7 Strassen products.
    # Let me define it as a list of 4 dicts: [c11_weights, c12_weights, c21_weights, c22_weights]

    sub_problems_corrected = [
        # (a_block, b_block, c_block, [c11, c12, c21, c22] weights)
        (0, 0, 0, [{0: 1, 3: 1, 4: -1, 6: 1}, {2: 1, 4: 1}, {1: 1, 3: 1}, {0: 1, 1: -1, 2: 1, 5: 1}]),
        (1, 2, 0, [{0: 1, 3: 1, 4: -1, 6: 1}, {2: 1, 4: 1}, {1: 1, 3: 1}, {0: 1, 1: -1, 2: 1, 5: 1}]),
        (0, 1, 1, [{0: 1, 3: 1, 4: -1, 6: 1}, {2: 1, 4: 1}, {1: 1, 3: 1}, {0: 1, 1: -1, 2: 1, 5: 1}]),
        (1, 3, 1, [{0: 1, 3: 1, 4: -1, 6: 1}, {2: 1, 4: 1}, {1: 1, 3: 1}, {0: 1, 1: -1, 2: 1, 5: 1}]),
    ]

    # Map local 2x2 indices to global indices
    # A is 2x4: row i (0,1), col j (0,1,2,3). A_block 0 uses cols 0-1, A_block 1 uses cols 2-3.
    # U index = i*m + j = i*4 + j
    def a_local_to_global(a_block, local_idx):
        # local_idx: 0=a11, 1=a12, 2=a21, 3=a22
        row = local_idx // 2
        col = local_idx % 2
        global_col = a_block * 2 + col
        return row * m + global_col

    # B is 4x4: row r (0-3), col c (0-3).
    # B_block 0: rows 0-1, cols 0-1
    # B_block 1: rows 0-1, cols 2-3
    # B_block 2: rows 2-3, cols 0-1
    # B_block 3: rows 2-3, cols 2-3
    def b_local_to_global(b_block, local_idx):
        # local_idx: 0=b11, 1=b12, 2=b21, 3=b22
        row = local_idx // 2
        col = local_idx % 2
        base_row = (b_block // 2) * 2  # 0 for blocks 0,1; 2 for blocks 2,3
        base_col = (b_block % 2) * 2   # 0 for blocks 0,2; 2 for blocks 1,3
        global_row = base_row + row
        global_col = base_col + col
        return global_row * p + global_col

    # C is 2x4: row i (0,1), col j (0-3).
    # C_block 0: cols 0-1, C_block 1: cols 2-3
    # W index follows W[k*n+i, r] where k is col of B (0-3), i is row of A (0-1), r is... 
    # Wait, the problem says W[k*n+i, r]. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # For 2x4 * 4x4 = 2x4 output.
    # U is (n*m, rank) = (8, rank), V is (m*p, rank) = (16, rank), W is (p*n, rank) = (8, rank).
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This gives C[i,k] = sum_j A[i,j] * B[j,k].
    # W index = k*n + i = k*2 + i.
    def c_local_to_global(c_block, local_idx):
        # local_idx: 0=c11, 1=c12, 2=c21, 3=c22
        row = local_idx // 2
        col = local_idx % 2
        global_col = c_block * 2 + col
        # W index = k*n + i where k is the B column (which becomes C column), i is A row
        # So W[k*n + i] = global_col * n + row
        return global_col * n + row

    # Now build U, V, W
    col = 0
    for a_block, b_block, c_block, c_weights in sub_problems_corrected:
        for s_idx, (a_coeffs, b_coeffs) in enumerate(strassen_products):
            # This Strassen product s_idx for this sub-problem
            # Add to U: for each A coefficient
            for local_a, weight_a in a_coeffs:
                if weight_a != 0:
                    u_idx = a_local_to_global(a_block, local_a)
                    U[u_idx][col] += weight_a
            # Add to V: for each B coefficient
            for local_b, weight_b in b_coeffs:
                if weight_b != 0:
                    v_idx = b_local_to_global(b_block, local_b)
                    V[v_idx][col] += weight_b
            # Add to W: for each of the 4 C output positions
            for c_pos in range(4):
                if s_idx in c_weights[c_pos]:
                    weight_c = c_weights[c_pos][s_idx]
                    if weight_c != 0:
                        w_idx = c_local_to_global(c_block, c_pos)
                        W[w_idx][col] += weight_c
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
