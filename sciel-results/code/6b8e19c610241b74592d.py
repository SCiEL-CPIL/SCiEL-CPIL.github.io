"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    # Block-diagonal decomposition: split the 3 intermediate rows of B into two
    # groups of sizes 1 and 2. The first group uses schoolbook (rank 2*10=20)
    # and the second group uses 2x2 Strassen on five disjoint 2-column pairs
    # of B (5*7=35). Total rank = 20 + 35 = 55.
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)]  # 20 x 55

    idx = 0

    # ---- Schoolbook block: j = 0 ----
    for i in range(2):
        for k in range(10):
            U_mat[i * 3 + 0][idx] = 1
            V_mat[0 * 10 + k][idx] = 1
            W_mat[k * 2 + i][idx] = 1
            idx += 1

    # ---- Strassen block: j = 1, 2 over five 2-column pairs of B ----
    # Within each 2x2 sub-problem:
    #   A' = [[A[0,1], A[0,2]], [A[1,1], A[1,2]]]
    #   B' = [[B[1,k0], B[1,k1]], [B[2,k0], B[2,k1]]]
    #   C' = C[:, {k0, k1}]
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1

        # U indices for A'
        a00 = 0 * 3 + 1  # A[0,1]
        a01 = 0 * 3 + 2  # A[0,2]
        a10 = 1 * 3 + 1  # A[1,1]
        a11 = 1 * 3 + 2  # A[1,2]

        # V indices for B'
        b0k0 = 1 * 10 + k0
        b0k1 = 1 * 10 + k1
        b1k0 = 2 * 10 + k0
        b1k1 = 2 * 10 + k1

        # W indices for C'
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1

        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] = 1
        U_mat[a11][idx] = 1
        V_mat[b0k0][idx] = 1
        V_mat[b1k1][idx] = 1
        W_mat[c0k0][idx] = 1
        W_mat[c1k1][idx] = 1
        idx += 1

        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] = 1
        U_mat[a11][idx] = 1
        V_mat[b0k0][idx] = 1
        W_mat[c1k0][idx] = 1
        W_mat[c1k1][idx] = -1
        idx += 1

        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] = 1
        V_mat[b0k1][idx] = 1
        V_mat[b1k1][idx] = -1
        W_mat[c0k1][idx] = 1
        W_mat[c1k1][idx] = 1
        idx += 1

        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] = 1
        V_mat[b1k0][idx] = 1
        V_mat[b0k0][idx] = -1
        W_mat[c0k0][idx] = 1
        W_mat[c1k0][idx] = 1
        idx += 1

        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] = 1
        U_mat[a01][idx] = 1
        V_mat[b1k1][idx] = 1
        W_mat[c0k0][idx] = -1
        W_mat[c0k1][idx] = 1
        idx += 1

        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] = 1
        U_mat[a00][idx] = -1
        V_mat[b0k0][idx] = 1
        V_mat[b0k1][idx] = 1
        W_mat[c1k1][idx] = 1
        idx += 1

        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] = 1
        U_mat[a11][idx] = -1
        V_mat[b1k0][idx] = 1
        V_mat[b1k1][idx] = 1
        W_mat[c0k0][idx] = 1
        idx += 1

    assert idx == rank, f"Expected rank {rank}, got {idx}"
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
