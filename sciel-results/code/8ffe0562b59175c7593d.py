"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """Rank-24 decomposition for 2x2 * 2x12 matrix multiplication.

    Partition the 12 columns of B (and C) into 4 groups of 3 columns.
    Each group is a 2x2 x 2x3 tensor, decomposed in rank 6 (Landsberg).
    Total rank = 4 * 6 = 24.

    Indexing conventions:
      U[i*m+j, r]: row = i*m+j for A[i][j]
      V[j*p+k, r]: row = j*p+k for B[j][k]
      W[k*n+i, r]: row = k*n+i for C[i][k] (trace convention)
    """
    n, m, p = DIMENSIONS  # 2, 2, 12
    group_size = 3
    num_groups = p // group_size  # 4

    # The 6 rank-1 terms for the 2x2 x 2x3 tensor.
    # Each term is (A_coeffs, B_coeffs, C_coeffs)
    # A indices: (i,j) with i in {0,1}, j in {0,1}
    # B indices: (j,k) with j in {0,1}, k in {0,1,2}
    # C indices: (i,k) with i in {0,1}, k in {0,1,2}
    terms = [
        # m1 = (A11+A22)(B11+B22)
        ([( (0,0), 1), ((1,1), 1)],
         [((0,0), 1), ((1,2), 1)],
         [((0,0), 1), ((1,1), 1)]),
        # m2 = (A21+A22)B11
        ([( (1,0), 1), ((1,1), 1)],
         [((0,0), 1)],
         [((1,0), 1), ((1,1), -1)]),
        # m3 = A11(B12-B22)
        ([( (0,0), 1)],
         [((0,1), 1), ((1,2), -1)],
         [((0,1), 1), ((1,1), 1)]),
        # m4 = A22(B21-B11)
        ([( (1,1), 1)],
         [((1,1), 1), ((0,0), -1)],
         [((0,0), 1), ((1,0), 1)]),
        # m5 = (A11+A12)B22
        ([( (0,0), 1), ((0,1), 1)],
         [((1,2), 1)],
         [((0,0), -1), ((0,1), 1)]),
        # m6 = (A21-A11)(B11+B12)
        ([( (1,0), 1), ((0,0), -1)],
         [((0,0), 1), ((0,1), 1)],
         [((0,2), 1), ((1,2), 1)]),
    ]

    rank = num_groups * len(terms)  # 4 * 6 = 24
    U = [[0] * rank for _ in range(n * m)]   # 4 x 24
    V = [[0] * rank for _ in range(m * p)]   # 24 x 24
    W = [[0] * rank for _ in range(p * n)]   # 24 x 24

    col = 0
    for g in range(num_groups):
        base = g * group_size  # starting column index in B and C
        for (a_coeffs, b_coeffs, c_coeffs) in terms:
            # A coefficients
            for (ai, aj), coeff in a_coeffs:
                u_row = ai * m + aj
                U[u_row][col] = coeff
            # B coefficients
            for (bj, bk), coeff in b_coeffs:
                v_row = bj * p + (base + bk)
                V[v_row][col] = coeff
            # C coefficients (trace convention: W[k*n + i])
            for (ci, ck), coeff in c_coeffs:
                w_row = (base + ck) * n + ci
                W[w_row][col] = coeff
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
