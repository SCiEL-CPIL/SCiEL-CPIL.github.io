"""Matrix multiplication 2x4 x 4x6 via 6 Strassen 2x2x2 blocks."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    # Strassen 2x2x2 factors: (A_coeffs, B_coeffs, C_coeffs)
    strassen_factors = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),  # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),                      # M2
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),                     # M3
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),           # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),           # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),           # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}), # M7
    ]

    a_col_pairs = [(0, 1), (2, 3)]
    b_col_pairs = [(0, 1), (2, 3), (4, 5)]

    factor_idx = 0
    for a_pair in a_col_pairs:
        j0, j1 = a_pair
        for b_pair in b_col_pairs:
            k0, k1 = b_pair
            for f in range(7):
                A_coeffs, B_coeffs, C_coeffs = strassen_factors[f]
                for (ai, aj), coeff in A_coeffs.items():
                    j_actual = j0 if aj == 0 else j1
                    U[ai * m + j_actual][factor_idx] += coeff
                for (bj, bk), coeff in B_coeffs.items():
                    j_actual = j0 if bj == 0 else j1
                    k_actual = k0 if bk == 0 else k1
                    V[j_actual * p + k_actual][factor_idx] += coeff
                for (ci, ck), coeff in C_coeffs.items():
                    i_actual = ci
                    k_actual = k0 if ck == 0 else k1
                    W[k_actual * n + i_actual][factor_idx] += coeff
                factor_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
