"""Matrix multiplication 2,2,13 — Strassen-block + schoolbook tail decomposition."""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13

    # Strategy: partition the 13 columns into 6 pairs of 2 (handled by Strassen,
    # rank 7 per block) plus 1 remaining column (schoolbook, rank 4).
    # Total rank = 6 * 7 + 4 = 46.

    rank = 46
    nU = n * m   # 4
    nV = m * p   # 26
    nW = p * n   # 26

    # Pre-allocate integer matrices
    U = [[0] * rank for _ in range(nU)]
    V = [[0] * rank for _ in range(nV)]
    W = [[0] * rank for _ in range(nW)]

    col = 0

    # Strassen pattern for a 2x2 x 2x2 block.
    # U indices: (i*m + j) -> 00:0, 01:1, 10:2, 11:3
    # V indices: (j*p + k) -> for local cols k0,k1: j=0 -> k, j=1 -> 13+k
    # W indices: (k*n + i) -> for local cols k0,k1: i=0 -> 2k, i=1 -> 2k+1

    # We define the 7 Strassen terms as tuples of (U_dict, V_dict, W_dict)
    # where each dict maps index -> coefficient.

    def strassen_terms(k0, k1):
        # Returns list of 7 (u, v, w) dicts
        terms = []
        # M1: (a00+a11)(b00+b11) -> contributes to c00, c11
        terms.append(({0: 1, 3: 1}, {k0: 1, 13 + k1: 1}, {2 * k0: 1, 2 * k1 + 1: 1}))
        # M2: (a10+a11)b00 -> contributes to c10, c11
        terms.append(({2: 1, 3: 1}, {k0: 1}, {2 * k0 + 1: 1, 2 * k1 + 1: -1}))
        # M3: a00(b01-b11) -> contributes to c01, c11
        terms.append(({0: 1}, {k1: 1, 13 + k1: -1}, {2 * k1: 1, 2 * k1 + 1: 1}))
        # M4: a11(b10-b00) -> contributes to c10, c00
        terms.append(({3: 1}, {13 + k0: 1, k0: -1}, {2 * k0: 1, 2 * k0 + 1: 1}))
        # M5: (a00+a01)b11 -> contributes to c01, c00
        terms.append(({0: 1, 1: 1}, {13 + k1: 1}, {2 * k0: -1, 2 * k1: 1}))
        # M6: (a10-a00)(b00+b01) -> contributes to c11
        terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2 * k1 + 1: 1}))
        # M7: (a01-a11)(b10+b11) -> contributes to c00
        terms.append(({1: 1, 3: -1}, {13 + k0: 1, 13 + k1: 1}, {2 * k0: 1}))
        return terms

    # Fill in Strassen blocks for 6 pairs
    for blk in range(6):
        k0 = 2 * blk
        k1 = 2 * blk + 1
        for u_t, v_t, w_t in strassen_terms(k0, k1):
            for idx, val in u_t.items():
                U[idx][col] += val
            for idx, val in v_t.items():
                V[idx][col] += val
            for idx, val in w_t.items():
                W[idx][col] += val
            col += 1

    # Fill in schoolbook for the last column k=12
    k = 12
    # c00 += a00 * b0k
    U[0][col] = 1
    V[k][col] = 1
    W[2 * k][col] = 1
    col += 1

    # c00 += a01 * b1k
    U[1][col] = 1
    V[13 + k][col] = 1
    W[2 * k][col] = 1
    col += 1

    # c10 += a10 * b0k
    U[2][col] = 1
    V[k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1

    # c10 += a11 * b1k  (actually c11)
    U[3][col] = 1
    V[13 + k][col] = 1
    W[2 * k + 1][col] = 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
