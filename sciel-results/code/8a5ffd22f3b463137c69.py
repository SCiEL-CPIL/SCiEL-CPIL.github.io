"""
Tensor decomposition for 2x4 * 4x8 matrix multiplication using block Strassen.

Strategy: Split the inner dimension j (0..3) into two groups of 2.
  C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
Each term is a 2x2 * 2x8 multiplication.
For 2x2 * 2x8, apply the rank-7 Strassen scheme to each of the 4 column-pairs of B.
Total rank: 2 * (7 * 4) = 56.

W index convention: W[k*n + i, r] where n=2, p=8.
"""

DIMENSIONS = [2, 4, 8]


def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Strassen 7-term scheme for 2x2 * 2x2:
    # A-side linear forms (indices into 2x2 A matrix, flattened row-major: a00=0, a01=1, a10=2, a11=3)
    # B-side linear forms (indices into 2x2 B matrix, flattened row-major: b00=0, b01=1, b10=2, b11=3)
    # W-side: how each product M_s contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    #
    # M0 = (a00+a11)(b00+b11) -> C00 += 1, C11 += 1
    # M1 = (a10+a11) b00      -> C10 += 1, C11 -= 1
    # M2 = a00 (b01-b11)      -> C01 += 1, C11 += 1
    # M3 = a11 (b10-b00)      -> C00 += 1, C10 += 1
    # M4 = (a00+a01) b11      -> C00 -= 1, C01 += 1
    # M5 = (a10-a00)(b00+b01) -> C11 += 1
    # M6 = (a01-a11)(b10+b11) -> C00 += 1

    # A-side forms: list of (row, col, coeff) for each of the 7 terms
    # A is 2x2, so indices are (i, j) with i in {0,1}, j in {0,1}
    A_forms = [
        [(0, 0, 1), (1, 1, 1)],       # s=0: a00 + a11
        [(1, 0, 1), (1, 1, 1)],       # s=1: a10 + a11
        [(0, 0, 1)],                   # s=2: a00
        [(1, 1, 1)],                   # s=3: a11
        [(0, 0, 1), (0, 1, 1)],       # s=4: a00 + a01
        [(1, 0, 1), (0, 0, -1)],      # s=5: a10 - a00
        [(0, 1, 1), (1, 1, -1)],      # s=6: a01 - a11
    ]

    # B-side forms: list of (row, col, coeff) for each of the 7 terms
    B_forms = [
        [(0, 0, 1), (1, 1, 1)],       # s=0: b00 + b11
        [(0, 0, 1)],                   # s=1: b00
        [(0, 1, 1), (1, 1, -1)],      # s=2: b01 - b11
        [(1, 0, 1), (0, 0, -1)],      # s=3: b10 - b00
        [(1, 1, 1)],                   # s=4: b11
        [(0, 0, 1), (0, 1, 1)],       # s=5: b00 + b01
        [(1, 0, 1), (1, 1, 1)],       # s=6: b10 + b11
    ]

    # W-side: for each Strassen term s, contributions to (C00, C01, C10, C11)
    # These are the coefficients for the 2x2 output block
    W_coeffs = [
        [1, 0, 0, 1],   # s=0: C00+=1, C11+=1
        [0, 0, 1, -1],  # s=1: C10+=1, C11-=1
        [0, 1, 0, 1],   # s=2: C01+=1, C11+=1
        [1, 0, 1, 0],   # s=3: C00+=1, C10+=1
        [-1, 1, 0, 0],  # s=4: C00-=1, C01+=1
        [0, 0, 0, 1],   # s=5: C11+=1
        [1, 0, 0, 0],   # s=6: C00+=1
    ]

    # Total rank: 2 (j-groups) * 4 (k-pairs) * 7 (Strassen terms) = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56

    t = 0
    for j_group in range(2):  # j_group 0: j in {0,1}, j_group 1: j in {2,3}
        for k_pair in range(4):  # k_pair 0: k in {0,1}, 1: {2,3}, 2: {4,5}, 3: {6,7}
            for s in range(7):  # Strassen step
                # U: A[i, j] where i in {0,1}, j = 2*j_group + local_j
                for (ai, aj, coeff) in A_forms[s]:
                    U[ai * m + (2 * j_group + aj)][t] += coeff

                # V: B[j, k] where j = 2*j_group + local_j, k = 2*k_pair + local_k
                for (bj, bk, coeff) in B_forms[s]:
                    V[(2 * j_group + bj) * p + (2 * k_pair + bk)][t] += coeff

                # W: C[i, k] where i in {0,1}, k = 2*k_pair + local_k
                # W index: W[k*n + i, r]
                # W_coeffs[s] = [C00_coeff, C01_coeff, C10_coeff, C11_coeff]
                c00, c01, c10, c11 = W_coeffs[s]
                if c00 != 0:
                    W[(2 * k_pair) * n + 0][t] += c00
                if c01 != 0:
                    W[(2 * k_pair + 1) * n + 0][t] += c01
                if c10 != 0:
                    W[(2 * k_pair) * n + 1][t] += c10
                if c11 != 0:
                    W[(2 * k_pair + 1) * n + 1][t] += c11

                t += 1

    return {"U": U, "V": V, "W": W}
