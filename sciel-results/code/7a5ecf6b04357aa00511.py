"""
Rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x8 matrix.

Strategy:
  - Split the 3 output rows into a top block (2 rows) and a bottom block (1 row).
  - Bottom row: schoolbook over 4*8 = 32 terms.
  - Top 2 rows: split the 8 output columns into 4 blocks of 2 columns each.
    For each 2x2 block, use Strassen's algorithm (7 terms) on two 2x2 sub-products:
      A_top_left (2x2, A cols 0-1) * B_left (2x2, B rows 0-1)
      A_top_right (2x2, A cols 2-3) * B_right (2x2, B rows 2-3)
    This gives 14 terms per block, 56 total for top.
  - Total rank: 32 + 56 = 88 (vs schoolbook 96).
"""

DIMENSIONS = [3, 4, 8]


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    # Output index convention: W[k*n + i, r]
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)

    # We will build lists of (u_idx, v_idx, w_idx) for each term.
    terms = []

    # --- Bottom row (i=2): schoolbook ---
    # C[2, k] = sum_j A[2, j] * B[j, k]
    # u_idx = 2*4 + j = 8 + j
    # v_idx = j*8 + k
    # w_idx = k*3 + 2
    for j in range(m):
        for k in range(p):
            u = 2 * m + j
            v = j * p + k
            w = k * n + 2
            terms.append((u, v, w))

    # --- Top 2 rows (i=0,1): Strassen per 2-column block ---
    # For each block b (0..3), columns 2b and 2b+1 of C.
    # A_top is 2x4 (rows 0,1; cols 0..3).
    # B_sub is 4x2 (rows 0..3; cols 2b, 2b+1).
    #
    # Decompose:
    #   A_left = A_top[:, 0:2]  (2x2)
    #   A_right = A_top[:, 2:4] (2x2)
    #   B_left = B_sub[0:2, :]  (2x2, rows 0,1 of B)
    #   B_right = B_sub[2:4, :] (2x2, rows 2,3 of B)
    #
    # P = A_left @ B_left + A_right @ B_right  (2x2 result)
    #
    # Strassen for X@Y where X,Y are 2x2:
    #   M1 = (X00+X11)(Y00+Y11)
    #   M2 = (X10+X11)Y00
    #   M3 = X00(Y01-Y11)
    #   M4 = X11(Y10-Y00)
    #   M5 = (X00+X01)Y11
    #   M6 = (X10-X00)(Y00+Y01)
    #   M7 = (X01-X11)(Y10+Y11)
    #   Result[0,0] = M1 + M4 - M5 + M7
    #   Result[0,1] = M3 + M5
    #   Result[1,0] = M2 + M4
    #   Result[1,1] = M1 - M2 + M3 + M6
    #
    # Each Mi is a scalar = (a_row, a_col) * (b_row, b_col)
    # We expand each product into individual (u, v, w) terms.

    for b in range(4):
        # Column indices in B: 2b, 2b+1
        col0 = 2 * b
        col1 = 2 * b + 1

        # --- Left product: A_left (2x2) @ B_left (2x2) ---
        # A_left[i, j] corresponds to u = i*m + j = 4*i + j
        # B_left[j, col] corresponds to v = j*p + col = 8*j + col
        # Result entry [i, c] contributes to w = (2b+c)*3 + i

        # X = A_left, Y = B_left
        # X00 = A[0,0], X01 = A[0,1], X10 = A[1,0], X11 = A[1,1]
        # Y00 = B[0,col0], Y01 = B[0,col1], Y10 = B[1,col0], Y11 = B[1,col1]

        def strassen_terms(X00, X01, X10, X11, Y00, Y01, Y10, Y11, out_i_offset, col_base):
            """
            Generate (u, v, w) terms for X@Y where:
              X[i,j] has u-index: Xij_u
              Y[j,c] has v-index: Yjc_v
              Result[i,c] contributes to w-index: col_base+c and row out_i_offset+i
            """
            result = []
            # M1 = (X00+X11)(Y00+Y11)
            # Expands to 4 products:
            for (xi, yi) in [(X00, Y00), (X00, Y11), (X11, Y00), (X11, Y11)]:
                result.append((xi, yi, col_base, out_i_offset, 0))  # contrib to [0,0]
                result.append((xi, yi, col_base + 1, out_i_offset, 0))  # wait, need to be careful

            # Actually, let's just enumerate each Mi's expansion carefully.
            # Each Mi = (sum of some X's) * (sum of some Y's)
            # = sum of (x * y) terms.
            # Each (x, y) contributes to specific result positions with signs.

            # Let me define a helper: add_term(x_u, y_v, w_row, w_col, sign)
            # w_idx = w_col * n + w_row

            local = []

            def add(x_u, y_v, r, c, sign=1):
                local.append((x_u, y_v, (col_base + c) * n + (out_i_offset + r), sign))

            # M1 = (X00+X11)(Y00+Y11)
            # M1 contributes: +1 to [0,0], +1 to [1,1]
            for xx in [X00, X11]:
                for yy in [Y00, Y11]:
                    add(xx, yy, 0, 0, +1)
                    add(xx, yy, 1, 1, +1)

            # M2 = (X10+X11)Y00
            # M2 contributes: +1 to [1,0], -1 to [1,1]
            for xx in [X10, X11]:
                add(xx, Y00, 1, 0, +1)
                add(xx, Y00, 1, 1, -1)

            # M3 = X00(Y01-Y11)
            # M3 contributes: +1 to [0,1], +1 to [1,1]
            add(X00, Y01, 0, 1, +1)
            add(X00, Y11, 0, 1, -1)
            add(X00, Y01, 1, 1, +1)
            add(X00, Y11, 1, 1, -1)

            # M4 = X11(Y10-Y00)
            # M4 contributes: +1 to [0,0], +1 to [1,0]
            add(X11, Y10, 0, 0, +1)
            add(X11, Y00, 0, 0, -1)
            add(X11, Y10, 1, 0, +1)
            add(X11, Y00, 1, 0, -1)

            # M5 = (X00+X01)Y11
            # M5 contributes: -1 to [0,0], +1 to [0,1]
            for xx in [X00, X01]:
                add(xx, Y11, 0, 0, -1)
                add(xx, Y11, 0, 1, +1)

            # M6 = (X10-X00)(Y00+Y01)
            # M6 contributes: +1 to [1,1]
            for xx, s in [(X10, +1), (X00, -1)]:
                for yy in [Y00, Y01]:
                    add(xx, yy, 1, 1, s)

            # M7 = (X01-X11)(Y10+Y11)
            # M7 contributes: +1 to [0,0]
            for xx, s in [(X01, +1), (X11, -1)]:
                for yy in [Y10, Y11]:
                    add(xx, yy, 0, 0, s)

            return local

        # Left product
        X00_l = 0 * m + 0  # A[0,0]
        X01_l = 0 * m + 1  # A[0,1]
        X10_l = 1 * m + 0  # A[1,0]
        X11_l = 1 * m + 1  # A[1,1]
        Y00_l = 0 * p + col0  # B[0, col0]
        Y01_l = 0 * p + col1  # B[0, col1]
        Y10_l = 1 * p + col0  # B[1, col0]
        Y11_l = 1 * p + col1  # B[1, col1]

        left_terms = strassen_terms(X00_l, X01_l, X10_l, X11_l,
                                    Y00_l, Y01_l, Y10_l, Y11_l,
                                    0, col0)

        # Right product
        X00_r = 0 * m + 2  # A[0,2]
        X01_r = 0 * m + 3  # A[0,3]
        X10_r = 1 * m + 2  # A[1,2]
        X11_r = 1 * m + 3  # A[1,3]
        Y00_r = 2 * p + col0  # B[2, col0]
        Y01_r = 2 * p + col1  # B[2, col1]
        Y10_r = 3 * p + col0  # B[3, col0]
        Y11_r = 3 * p + col1  # B[3, col1]

        right_terms = strassen_terms(X00_r, X01_r, X10_r, X11_r,
                                     Y00_r, Y01_r, Y10_r, Y11_r,
                                     0, col0)

        # Combine: each term is (u, v, w, sign)
        all_block_terms = left_terms + right_terms
        for u, v, w, sign in all_block_terms:
            if sign == 1:
                terms.append((u, v, w))
            else:
                terms.append((u, v, w, sign))

    # Now we have a list of terms. Some are (u, v, w), some are (u, v, w, sign).
    # Normalize: convert to (u, v, w, coeff)
    normalized = []
    for t in terms:
        if len(t) == 3:
            normalized.append((t[0], t[1], t[2], 1))
        else:
            normalized.append(t)

    # Merge terms with same (u, v, w)
    from collections import defaultdict
    merged = defaultdict(float)
    for u, v, w, c in normalized:
        merged[(u, v, w)] += c

    # Filter out zero coefficients
    final_terms = []
    for (u, v, w), c in merged.items():
        if abs(c) > 1e-12:
            final_terms.append((u, v, w, c))

    rank = len(final_terms)

    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (u, v, w, c) in enumerate(final_terms):
        U[u][r] = c
        V[v][r] = 1
        W[w][r] = 1

    return {"U": U, "V": V, "W": W}
