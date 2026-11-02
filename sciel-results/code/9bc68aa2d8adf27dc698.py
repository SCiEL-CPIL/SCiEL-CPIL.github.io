"""Rewritten program: structured block decomposition using Strassen for 3x4x4 matrix multiplication."""

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    dim_U = n * m  # 12
    dim_V = m * p  # 16
    dim_W = p * n  # 12

    # We'll build the decomposition using block matrix multiplication.
    # A is 3x4, B is 4x4, C is 3x4.
    # Split A into rows: A_0 (1x4), A_1 (1x4), A_2 (1x4)
    # Split B into columns: B_0 (4x1), B_1 (4x1), B_2 (4x1), B_3 (4x1)
    # C_i = A_i * B (full 4x4 B)
    # Actually, let's think of it as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Let's use a different decomposition approach.
    # We can write the 3x4x4 multiplication as a sum of smaller multiplications.
    # 
    # Strategy: Use the fact that 3 = 2 + 1, 4 = 2 + 2.
    # Split A (3x4) into:
    #   A_top (2x4), A_bot (1x4)
    # Split B (4x4) into:
    #   B_left (4x2), B_right (4x2)
    # 
    # C_top (2x4) = A_top * B (2x4 * 4x4 = 2x4)
    # C_bot (1x4) = A_bot * B (1x4 * 4x4 = 1x4)
    #
    # For A_top * B (2x4 * 4x4):
    #   Split A_top (2x4) into A_tl (2x2), A_tr (2x2)
    #   Split B (4x4) into B_tl (2x2), B_tr (2x2), B_bl (2x2), B_br (2x2)
    #   C_tl (2x2) = A_tl * B_tl + A_tr * B_bl
    #   C_tr (2x2) = A_tl * B_tr + A_tr * B_br
    #   Each 2x2*2x2 can be done with Strassen (7 mults)
    #   So C_top uses 4 * 7 = 28 mults
    #
    # For A_bot * B (1x4 * 4x4):
    #   This is just a row vector times a matrix.
    #   C_bot[k] = sum_j A_bot[j] B[j,k]
    #   We can split B into columns and use standard multiplication.
    #   1x4 * 4x4 = 1x4, which requires 16 scalar multiplications.
    #   But we can do better by splitting into 1x2 * 2x4 blocks.
    #   Actually, 1x4 * 4x4: split A_bot into (1x2, 1x2), B into (2x4, 2x4)
    #   C_bot = A_bot_left * B_top + A_bot_right * B_bot
    #   Each 1x2 * 2x4 = 1x4 requires 8 scalar mults.
    #   Total: 16 scalar mults.
    #
    # Total: 28 + 16 = 44 mults.
    #
    # Now I need to map each scalar multiplication to a (u, v, w) triple.
    # For a scalar multiplication A[i,j] * B[j,k], the corresponding triple has:
    #   u[i*m + j] = 1, v[j*p + k] = 1, w[k*n + i] = 1, all other entries 0.
    # For Strassen, each "multiplication" is of the form (sum of A terms) * (sum of B terms).
    # E.g., M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This corresponds to u having 1s at positions for A[0,0] and A[1,1],
    # v having 1s at positions for B[0,0] and B[1,1],
    # and w having 1 at the position corresponding to the output.
    # But wait, the output of Strassen is a combination, not a single output.
    # Actually, each Strassen multiplication produces one intermediate value,
    # and that value contributes to multiple output entries.
    #
    # Let me reconsider. In the tensor decomposition, each rank-1 term is:
    #   u ⊗ v ⊗ w
    # where u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}.
    # The contribution to the tensor T[i,j,k] is u[i*m+j] * v[j*p+k] * w[k*n+i].
    #
    # For a standard multiplication A[i,j]*B[j,k] contributing to C[i,k]:
    #   u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1.
    #
    # For a Strassen multiplication like M = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2]):
    #   This is actually 4 scalar multiplications:
    #   A[i1,j1]*B[j1,k1] + A[i1,j1]*B[j2,k2] + A[i2,j2]*B[j1,k1] + A[i2,j2]*B[j2,k2]
    #   But in Strassen, we count this as ONE multiplication because we compute the sum first.
    #   In the tensor decomposition, this corresponds to:
    #   u has 1s at (i1,j1) and (i2,j2)
    #   v has 1s at (j1,k1) and (j2,k2)
    #   w has 1 at (k_out, i_out) where the result contributes.
    #   But the result M contributes to multiple C entries with coefficients.
    #
    # Actually, I think the correct way is:
    # Each Strassen multiplication M produces a scalar value.
    # That scalar value is then used in linear combinations to form C entries.
    # In the tensor decomposition, each such multiplication corresponds to ONE rank-1 term,
    # where u and v capture the sums, and w captures which output entries this M contributes to,
    # with the appropriate coefficients.
    #
    # Let me implement this carefully.
    #
    # I'll define a helper to add a rank-1 term.
    # Each term is specified by:
    #   u_dict: dict from (i,j) to coefficient
    #   v_dict: dict from (j,k) to coefficient
    #   w_dict: dict from (k,i) to coefficient
    #
    # The contribution to T[i,j,k] is sum over all terms of u_dict[(i,j)] * v_dict[(j,k)] * w_dict[(k,i)].
    # We need this to equal 1 for all (i,j,k) where the multiplication A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, T[i,j,k] = 1 for all valid i,j,k.

    # Let me use a list of (u, v, w) triples, each as dicts.
    terms = []

    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    # Strassen 2x2x2:
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For a 2x2 block of A at rows (r1,r2) and cols (c1,c2),
    # and a 2x2 block of B at rows (c1,c2) and cols (d1,d2),
    # the Strassen multiplications are:

    def strassen_2x2(a_rows, a_cols, b_rows, b_cols, w_offset):
        """
        a_rows: 2 row indices in A
        a_cols: 2 col indices in A (which are row indices in B)
        b_rows: should be same as a_cols
        b_cols: 2 col indices in B
        w_offset: not used, we determine w from context
        """
        a_r1, a_r2 = a_rows
        a_c1, a_c2 = a_cols
        b_c1, b_c2 = b_cols

        # M1 = (A[a_r1,a_c1] + A[a_r2,a_c2]) * (B[a_c1,b_c1] + B[a_c2,b_c2])
        u1 = {(a_r1, a_c1): 1, (a_r2, a_c2): 1}
        v1 = {(a_c1, b_c1): 1, (a_c2, b_c2): 1}
        # This M1 contributes to C[a_r1,b_c1] with +1, C[a_r1,b_c2] with -1,
        # C[a_r2,b_c1] with +1, C[a_r2,b_c2] with -1
        # Wait, let me re-check Strassen formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes: C11 += M1, C22 += M1
        # In terms of output indices:
        # C[a_r1,b_c1] += M1, C[a_r2,b_c2] += M1
        w1 = {(b_c1, a_r1): 1, (b_c2, a_r2): 1}
        add_term(u1, v1, w1)

        # M2 = (A[a_r2,a_c1] + A[a_r2,a_c2]) * B[a_c1,b_c1]
        u2 = {(a_r2, a_c1): 1, (a_r2, a_c2): 1}
        v2 = {(a_c1, b_c1): 1}
        # M2 contributes: C21 += M2, C22 -= M2
        w2 = {(b_c1, a_r2): 1, (b_c2, a_r2): -1}
        add_term(u2, v2, w2)

        # M3 = A[a_r1,a_c1] * (B[a_c1,b_c2] - B[a_c2,b_c2])
        u3 = {(a_r1, a_c1): 1}
        v3 = {(a_c1, b_c2): 1, (a_c2, b_c2): -1}
        # M3 contributes: C12 += M3, C22 += M3
        w3 = {(b_c2, a_r1): 1, (b_c2, a_r2): 1}
        add_term(u3, v3, w3)

        # M4 = A[a_r2,a_c2] * (B[a_c2,b_c1] - B[a_c1,b_c1])
        u4 = {(a_r2, a_c2): 1}
        v4 = {(a_c2, b_c1): 1, (a_c1, b_c1): -1}
        # M4 contributes: C11 += M4, C21 += M4
        w4 = {(b_c1, a_r1): 1, (b_c1, a_r2): 1}
        add_term(u4, v4, w4)

        # M5 = (A[a_r1,a_c1] + A[a_r1,a_c2]) * B[a_c2,b_c2]
        u5 = {(a_r1, a_c1): 1, (a_r1, a_c2): 1}
        v5 = {(a_c2, b_c2): 1}
        # M5 contributes: C11 -= M5, C12 += M5
        w5 = {(b_c1, a_r1): -1, (b_c2, a_r1): 1}
        add_term(u5, v5, w5)

        # M6 = (A[a_r2,a_c1] - A[a_r1,a_c1]) * (B[a_c1,b_c1] + B[a_c1,b_c2])
        u6 = {(a_r2, a_c1): 1, (a_r1, a_c1): -1}
        v6 = {(a_c1, b_c1): 1, (a_c1, b_c2): 1}
        # M6 contributes: C22 += M6
        w6 = {(b_c2, a_r2): 1}
        add_term(u6, v6, w6)

        # M7 = (A[a_r1,a_c2] - A[a_r2,a_c2]) * (B[a_c2,b_c1] + B[a_c2,b_c2])
        u7 = {(a_r1, a_c2): 1, (a_r2, a_c2): -1}
        v7 = {(a_c2, b_c1): 1, (a_c2, b_c2): 1}
        # M7 contributes: C11 += M7
        w7 = {(b_c1, a_r1): 1}
        add_term(u7, v7, w7)

    # Now build the full decomposition.
    # A is 3x4, B is 4x4.
    # Split A into A_top (rows 0,1) and A_bot (row 2).
    # A_top is 2x4, A_bot is 1x4.
    # B is 4x4, split into B_top (rows 0,1) and B_bot (rows 2,3).
    #
    # C_top (2x4) = A_top * B
    #   A_top (2x4) * B (4x4)
    #   Split A_top into A_tl (2x2, cols 0,1) and A_tr (2x2, cols 2,3)
    #   Split B into B_tl (2x2, rows 0,1 cols 0,1), B_tr (2x2, rows 0,1 cols 2,3),
    #              B_bl (2x2, rows 2,3 cols 0,1), B_br (2x2, rows 2,3 cols 2,3)
    #   C_tl (2x2) = A_tl * B_tl + A_tr * B_bl
    #   C_tr (2x2) = A_tl * B_tr + A_tr * B_br
    #
    # Each 2x2*2x2 uses Strassen (7 terms).
    # 4 such products: 4*7 = 28 terms.

    # C_top left half (cols 0,1):
    strassen_2x2((0,1), (0,1), (0,1), (0,1))  # A_tl * B_tl
    strassen_2x2((0,1), (2,3), (2,3), (0,1))  # A_tr * B_bl

    # C_top right half (cols 2,3):
    strassen_2x2((0,1), (0,1), (0,1), (2,3))  # A_tl * B_tr
    strassen_2x2((0,1), (2,3), (2,3), (2,3))  # A_tr * B_br

    # C_bot (1x4) = A_bot * B
    # A_bot is row 2, cols 0,1,2,3.
    # B is 4x4.
    # C_bot[k] = sum_j A[2,j] * B[j,k]
    # This is 16 scalar multiplications.
    # We can split into two 1x2 * 2x4 products.
    # A_bot_left (1x2, cols 0,1) * B_top (2x4, rows 0,1)
    # A_bot_right (1x2, cols 2,3) * B_bot (2x4, rows 2,3)
    #
    # For 1x2 * 2x4:
    # A = [a1, a2], B is 2x4.
    # C[k] = a1*B[0,k] + a2*B[1,k]
    # This is just 8 scalar multiplications (2x4).
    # Each is a standard rank-1 term.

    # A_bot_left * B_top:
    for j in range(2):  # j=0,1
        for k in range(4):  # k=0,1,2,3
            u = {(2, j): 1}
            v = {(j, k): 1}
            w = {(k, 2): 1}
            add_term(u, v, w)

    # A_bot_right * B_bot:
    for j in range(2, 4):  # j=2,3
        for k in range(4):  # k=0,1,2,3
            u = {(2, j): 1}
            v = {(j, k): 1}
            w = {(k, 2): 1}
            add_term(u, v, w)

    # Total terms: 28 + 8 + 8 = 44.
    rank = len(terms)

    # Now build U, V, W matrices.
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]

    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val

    # Verify the decomposition.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-9:
                    raise ValueError(f"Decomposition failed at ({i},{j},{k}): got {total}")

    return {"U": U, "V": V, "W": W}
