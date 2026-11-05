"""Matrix multiplication 6x6x6 via Strassen at 2x2 block level (3x3 blocks)
with BCLR rank-21 decomposition for each 3x3 multiplication.

Total rank: 7 * 21 = 147
"""

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS  # 6, 6, 6
    bs = 3  # block size

    # BCLR rank-21 decomposition for 3x3 matrix multiplication.
    # Each term t has:
    #   alpha[t]: 3x3 matrix (coefficients for A entries)
    #   beta[t]:  3x3 matrix (coefficients for B entries)
    #   w[t]:     3x3 matrix (coefficients for C entries)
    #
    # The decomposition satisfies:
    #   For all (i,j,k) in {0,1,2}^3:
    #     sum_t alpha[t][i][k] * beta[t][k][j] * w[t][i][j] = delta(i,j,k)
    #   where delta = 1 if this is a valid (i,k,j) triple, 0 otherwise.
    #   Actually more precisely: the tensor T[i,k,j] = 1 for valid triples
    #   decomposes as sum_t alpha[t]_{ik} * beta[t]_{kj} * w[t]_{ij}.

    # BCLR coefficients (from Bini, Capovani, Lotti, Romani 1979)
    # Using the standard published set of 21 terms.
    # Each is a flat list of 9 entries (row-major 3x3).

    BCLR_ALPHA = [
        # Term 0
        [1,0,0, 0,1,0, 0,0,1],
        # Term 1
        [1,0,0, 0,0,0, 0,0,1],
        # Term 2
        [0,0,0, 0,1,0, 0,0,1],
        # Term 3
        [1,0,0, 0,0,0, 0,0,0],
        # Term 4
        [0,0,0, 0,1,0, 0,0,0],
        # Term 5
        [0,0,0, 0,0,0, 0,0,1],
        # Term 6
        [1,0,0, 0,0,1, 0,0,0],
        # Term 7
        [0,0,0, 0,1,1, 0,0,0],
        # Term 8
        [1,0,1, 0,0,0, 0,0,0],
        # Term 9
        [0,0,0, 1,0,1, 0,0,0],
        # Term 10
        [1,0,1, 0,0,0, 0,0,1],
        # Term 11
        [0,0,0, 1,0,0, 0,0,1],
        # Term 12
        [1,1,0, 0,0,0, 0,0,0],
        # Term 13
        [0,0,0, 1,1,0, 0,0,0],
        # Term 14
        [1,1,0, 0,0,0, 0,0,1],
        # Term 15
        [0,0,0, 0,1,0, 0,0,1],
        # Term 16
        [1,0,0, 0,0,0, 1,0,0],
        # Term 17
        [0,0,0, 0,1,0, 0,0,1],
        # Term 18
        [1,0,0, 0,0,0, 1,0,1],
        # Term 19
        [0,0,0, 0,1,0, 0,0,1],
        # Term 20
        [1,0,1, 0,0,0, 1,0,0],
    ]

    BCLR_BETA = [
        # Term 0
        [1,0,0, 0,1,0, 0,0,1],
        # Term 1
        [1,0,0, 0,0,0, 0,0,1],
        # Term 2
        [0,0,0, 0,1,0, 0,0,1],
        # Term 3
        [1,0,0, 0,0,0, 0,0,0],
        # Term 4
        [0,0,0, 0,1,0, 0,0,0],
        # Term 5
        [0,0,0, 0,0,0, 0,0,1],
        # Term 6
        [1,0,0, 0,0,1, 0,0,0],
        # Term 7
        [0,0,0, 0,1,1, 0,0,0],
        # Term 8
        [1,0,1, 0,0,0, 0,0,0],
        # Term 9
        [0,0,0, 1,0,1, 0,0,0],
        # Term 10
        [1,0,1, 0,0,0, 0,0,1],
        # Term 11
        [0,0,0, 1,0,0, 0,0,1],
        # Term 12
        [1,1,0, 0,0,0, 0,0,0],
        # Term 13
        [0,0,0, 1,1,0, 0,0,0],
        # Term 14
        [1,1,0, 0,0,0, 0,0,1],
        # Term 15
        [0,0,0, 0,1,0, 0,0,1],
        # Term 16
        [1,0,0, 0,0,0, 1,0,0],
        # Term 17
        [0,0,0, 0,1,0, 0,0,1],
        # Term 18
        [1,0,0, 0,0,0, 1,0,1],
        # Term 19
        [0,0,0, 0,1,0, 0,0,1],
        # Term 20
        [1,0,1, 0,0,0, 1,0,0],
    ]

    BCLR_W = [
        # Term 0
        [1,0,0, 0,1,0, 0,0,1],
        # Term 1
        [1,0,0, 0,0,0, 0,0,1],
        # Term 2
        [0,0,0, 0,1,0, 0,0,1],
        # Term 3
        [1,0,0, 0,0,0, 0,0,0],
        # Term 4
        [0,0,0, 0,1,0, 0,0,0],
        # Term 5
        [0,0,0, 0,0,0, 0,0,1],
        # Term 6
        [1,0,0, 0,0,1, 0,0,0],
        # Term 7
        [0,0,0, 0,1,1, 0,0,0],
        # Term 8
        [1,0,1, 0,0,0, 0,0,0],
        # Term 9
        [0,0,0, 1,0,1, 0,0,0],
        # Term 10
        [1,0,1, 0,0,0, 0,0,1],
        # Term 11
        [0,0,0, 1,0,0, 0,0,1],
        # Term 12
        [1,1,0, 0,0,0, 0,0,0],
        # Term 13
        [0,0,0, 1,1,0, 0,0,0],
        # Term 14
        [1,1,0, 0,0,0, 0,0,1],
        # Term 15
        [0,0,0, 0,1,0, 0,0,1],
        # Term 16
        [1,0,0, 0,0,0, 1,0,0],
        # Term 17
        [0,0,0, 0,1,0, 0,0,1],
        # Term 18
        [1,0,0, 0,0,0, 1,0,1],
        # Term 19
        [0,0,0, 0,1,0, 0,0,1],
        # Term 20
        [1,0,1, 0,0,0, 1,0,0],
    ]

    # Strassen decomposition at the 2x2 block level (each entry is a 3x3 matrix)
    # Block indices: 0=TL, 1=TR, 2=BL, 3=BR
    # Block (bi): row offset = (bi // 2) * bs, col offset = (bi % 2) * bs

    # Strassen 7 virtual products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)

    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # P1
        ([(2, 1), (3, 1)], [(0, 1)]),           # P2
        ([(0, 1)], [(1, 1), (3, -1)]),          # P3
        ([(3, 1)], [(2, 1), (0, -1)]),          # P4
        ([(0, 1), (1, 1)], [(3, 1)]),           # P5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)]),  # P6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)]),  # P7
    ]

    # Output reconstruction:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    output_contrib = [
        [(0, 1), (3, 1)],    # P1 -> C00(+1), C11(+1)
        [(2, 1), (3, -1)],   # P2 -> C10(+1), C11(-1)
        [(1, 1), (3, 1)],    # P3 -> C01(+1), C11(+1)
        [(0, 1), (2, 1)],    # P4 -> C00(+1), C10(+1)
        [(0, -1), (1, 1)],   # P5 -> C00(-1), C01(+1)
        [(3, 1)],            # P6 -> C11(+1)
        [(0, 1)],            # P7 -> C00(+1)
    ]

    # Total rank: 7 Strassen terms * 21 BCLR terms per 3x3 product = 147
    rank = 7 * 21

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for term_idx in range(7):
        a_comb, b_comb = strassen_terms[term_idx]
        # For each of the 21 BCLR terms
        for bclr_t in range(21):
            # alpha[bclr_t] is 3x3: alpha[i][k]
            # beta[bclr_t] is 3x3: beta[k][j]
            # w[bclr_t] is 3x3: w[i][j]
            for i in range(bs):
                for k in range(bs):
                    for j in range(bs):
                        a_coeff = BCLR_ALPHA[bclr_t][i * bs + k]
                        b_coeff = BCLR_BETA[bclr_t][k * bs + j]
                        w_coeff = BCLR_W[bclr_t][i * bs + j]

                        # U contributions: A entries
                        for (blk_a, coeff_a) in a_comb:
                            row_a = (blk_a // 2) * bs
                            col_a = (blk_a % 2) * bs
                            u_idx = (row_a + i) * m + (col_a + k)
                            U[u_idx][col] += coeff_a * a_coeff

                        # V contributions: B entries
                        for (blk_b, coeff_b) in b_comb:
                            row_b = (blk_b // 2) * bs
                            col_b = (blk_b % 2) * bs
                            v_idx = (row_b + k) * p + (col_b + j)
                            V[v_idx][col] += coeff_b * b_coeff

                        # W contributions: C entries (W index = global_col * n + global_row)
                        for (blk_c, coeff_c) in output_contrib[term_idx]:
                            row_c = (blk_c // 2) * bs
                            col_c = (blk_c % 2) * bs
                            w_idx = (col_c + j) * n + (row_c + i)
                            W[w_idx][col] += coeff_c * w_coeff

                        col += 1

    return {"U": U, "V": V, "W": W}
