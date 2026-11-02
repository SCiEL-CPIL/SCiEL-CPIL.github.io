"""
Rank-56 exact decomposition for 2x4 x 4x8 matrix multiplication.

Uses Strassen's 7-multiplication algorithm on each 2x2 x 2x2 block.
The 2x4 matrix is split into 2 column-pairs, each multiplied by 4 row-pairs
of the 4x8 matrix, giving 8 independent 2x2x2 blocks.

Rank = 8 blocks * 7 Strassen multiplications = 56.
"""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = [[0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0] * rank for _ in range(p * n)]   # 16 x 56

    # Strassen decomposition of 2x2x2 matrix multiplication.
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A[local] * B[local] -> C[local]
    # Each step: (U_local_coeffs, V_local_coeffs, W_local_coeffs)
    strassen_steps = [
        # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00*(B11-B01) -> C01 += M3, C11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)*B11 -> C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)*(B00+B01) -> C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)*(B10+B11) -> C00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    col = 0
    # h: which column-pair of A (and row-pair of B) we're in: 0, 1
    # c: which row-pair of B (and column-pair of C) we're in: 0, 1, 2, 3
    for h in range(2):
        for c in range(4):
            for s in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen_steps[s]

                # Fill U: row index = i * m + j_global, where j_global = 2*h + j_local
                for local_idx, val in u_coeffs.items():
                    i = local_idx // 2
                    j_l = local_idx % 2
                    j_g = 2 * h + j_l
                    U[i * m + j_g][col] += val

                # Fill V: row index = j_global * p + k_global, where j_global = 2*h + j_local, k_global = 2*c + k_local
                for local_idx, val in v_coeffs.items():
                    j_l = local_idx // 2
                    k_l = local_idx % 2
                    j_g = 2 * h + j_l
                    k_g = 2 * c + k_l
                    V[j_g * p + k_g][col] += val

                # Fill W: row index = k_global * n + i, where k_global = 2*c + k_local
                for local_idx, val in w_coeffs.items():
                    i = local_idx // 2
                    k_l = local_idx % 2
                    k_g = 2 * c + k_l
                    W[k_g * n + i][col] += val

                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
