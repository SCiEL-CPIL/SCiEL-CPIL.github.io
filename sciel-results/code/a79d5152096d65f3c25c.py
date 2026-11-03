"""Matrix multiplication 3,4,4 rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-44 exact tensor decomposition for 3x4 by 4x4 matrix multiplication.
    Uses Strassen's algorithm (rank 7) for each of the four 2x2x2 blocks covering
    A[0:2, :], B[:, 0:2], and C[0:2, :], and schoolbook (rank 16) for the remaining
    row i=2.
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank for _ in range(p * n)]  # 12 x 44

    # Strassen's 7 products for 2x2 matrices:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B10)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A10)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    #
    # C = [[M1 - M4 + M5 + M7, M3 + M4],
    #      [M2 - M6 - M7,     M1 - M2 + M3 + M6]]

    # For a 2x2x2 block at offsets (i0, j0, k0):
    # U indices: A local (ii, jj) -> global (i0+ii, j0+jj) -> U[(i0+ii)*m + (j0+jj)]
    # V indices: B local (jj, kk) -> global (j0+jj, k0+kk) -> V[(j0+jj)*p + (k0+kk)]
    # W indices: C local (kk, ii) -> global (k0+kk, i0+ii) -> W[(k0+kk)*n + (i0+ii)]

    # Strassen term definitions: (u_coeffs, v_coeffs, w_coeffs)
    # Each is a list of (local_idx, coeff) for the 4 local elements.
    # Local A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Local B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Local C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3

    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ([((0, 1), (3, 1))], [((0, 1), (3, 1))], [((0, 1), (3, 1))]),
        # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2
        ([((2, 1), (3, 1))], [((0, 1))], [((2, 1), (3, -1))]),
        # M3: A00(B10-B11) -> C01 += M3, C11 += M3
        ([((0, 1))], [((2, 1), (3, -1))], [((1, 1), (3, 1))]),
        # M4: A11(B01-B10) -> C00 -= M4, C01 += M4
        ([((3, 1))], [((1, 1), (2, -1))], [((0, -1), (1, 1))]),
        # M5: (A00+A10)B11 -> C00 += M5
        ([((0, 1), (2, 1))], [((3, 1))], [((0, 1))]),
        # M6: (A11-A10)(B00+B10) -> C10 -= M6, C11 += M6
        ([((3, 1), (2, -1))], [((0, 1), (2, 1))], [((2, -1), (3, 1))]),
        # M7: (A01-A11)(B01+B11) -> C00 += M7, C10 -= M7
        ([((1, 1), (3, -1))], [((1, 1), (3, 1))], [((0, 1), (2, -1))]),
    ]

    col = 0
    for i0 in range(0, n, 2):
        for j0 in range(0, m, 2):
            for k0 in range(0, p, 2):
                # Check if this is a valid 2x2x2 block
                if i0 + 2 > n or j0 + 2 > m or k0 + 2 > p:
                    continue
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    for li, coeff_u in u_coeffs:
                        ii = li // 2
                        jj = li % 2
                        U[(i0 + ii) * m + (j0 + jj)][col] += coeff_u
                    for li, coeff_v in v_coeffs:
                        jj = li // 2
                        kk = li % 2
                        V[(j0 + jj) * p + (k0 + kk)][col] += coeff_v
                    for li, coeff_w in w_coeffs:
                        kk = li // 2
                        ii = li % 2
                        W[(k0 + kk) * n + (i0 + ii)][col] += coeff_w
                    col += 1

    # Schoolbook terms for i=2 (the third row of A and C)
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
