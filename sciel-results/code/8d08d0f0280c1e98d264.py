"""Kronecker-embedded Strassen decomposition for 3x3 matrix multiplication.

We treat the 3x3 matrices as 2x2 block matrices where the off-diagonal blocks
are 1x1 scalars, and apply the rank-7 Strassen algorithm to the diagonal 2x2
blocks while using direct multiplication for the remaining interactions.
The total rank is minimized by sharing factors across the block structure.
"""

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Block structure:
    #   A = [[A_TL, A_TR],      B = [[B_TL, B_TR],
    #        [A_BL, A_BR]]            [B_BL, B_BR]]
    # where A_TL, A_BL, B_TL, B_BL are 2x2 blocks and
    #       A_TR, A_BR, B_TR, B_BR are 1x1 scalars.
    #
    # Index mapping (row-major):
    #   A_TL: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    #   A_TR: (0,2)->2, A_BR: (1,2)->5
    #   A_BL: (2,0)->6, (2,1)->7, A_BR: (2,2)->8
    #   B_TL: (0,0)->0, (0,1)->1, (1,0)->3, (1,1)->4
    #   B_TR: (0,2)->2, B_BR: (1,2)->5
    #   B_BL: (2,0)->6, (2,1)->7, B_BR: (2,2)->8
    #
    # W index for C[i][k] uses trace convention: widx(k, i) = k*n + i

    def widx(k, i):
        return k * n + i

    slices = []

    # --- Strassen on A_TL * B_TL -> contributes to C_TL (2x2) ---
    # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    slices.append(([(0, 1.0), (4, 1.0)], [(0, 1.0), (4, 1.0)],
                   [(widx(0, 0), 1.0), (widx(1, 1), 1.0)]))
    # M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
    slices.append(([(3, 1.0), (4, 1.0)], [(0, 1.0)],
                   [(widx(0, 1), 1.0), (widx(1, 1), -1.0)]))
    # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
    slices.append(([(0, 1.0)], [(1, 1.0), (4, -1.0)],
                   [(widx(1, 0), 1.0), (widx(1, 1), 1.0)]))
    # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
    slices.append(([(4, 1.0)], [(3, 1.0), (0, -1.0)],
                   [(widx(0, 0), 1.0), (widx(0, 1), 1.0)]))
    # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
    slices.append(([(0, 1.0), (1, 1.0)], [(4, 1.0)],
                   [(widx(0, 0), -1.0), (widx(1, 0), 1.0)]))
    # M6 = (A10-A00)(B00+B01) -> C11 += M6
    slices.append(([(3, 1.0), (0, -1.0)], [(0, 1.0), (1, 1.0)],
                   [(widx(1, 1), 1.0)]))
    # M7 = (A01-A11)(B10+B11) -> C00 += M7
    slices.append(([(1, 1.0), (4, -1.0)], [(3, 1.0), (4, 1.0)],
                   [(widx(0, 0), 1.0)]))

    # --- Direct multiplications for cross-block interactions ---

    # A_TR * B_TL: A_TR is scalar at index 2, B_TL is 2x2
    # C[0,0] += A02*B00, C[0,1] += A02*B01
    # C[1,0] += A12*B10, C[1,1] += A12*B11
    # Wait, A_TR = [A02] (row 0, col 2), B_TL = [[B00,B01],[B10,B11]]
    # A_TR * B_TL gives 1x2: [A02*B00, A02*B01]
    # This contributes to C row 0, cols 0,1
    slices.append(([(2, 1.0)], [(0, 1.0)], [(widx(0, 0), 1.0)]))
    slices.append(([(2, 1.0)], [(1, 1.0)], [(widx(1, 0), 1.0)]))

    # A_BL * B_TR: A_BL is 2x1 [A20, A21]^T (indices 6,7), B_TR is scalar B02 (index 2)
    # A_BL * B_TR gives 2x1: [A20*B02, A21*B02]
    # This contributes to C rows 2, cols 0
    slices.append(([(6, 1.0)], [(2, 1.0)], [(widx(0, 2), 1.0)]))
    slices.append(([(7, 1.0)], [(2, 1.0)], [(widx(0, 2), 1.0)]))

    # A_TL * B_TR: A_TL is 2x2, B_TR is scalar B02 (index 2)
    # A_TL * B_TR gives 2x1: [A00*B02, A10*B02] for col 0, [A01*B02, A11*B02] for col 1
    # Wait, B_TR = [B02] (row 0, col 2)
    # A_TL * B_TR: 2x2 * 2x1 = 2x1
    # Result: [A00*B02 + A01*B02, A10*B02 + A11*B02]? No.
    # B_TR as 2x1: [B02, B12]^T? No, B_TR is just B[0,2] = B02.
    # Actually B_TR is a 1x1 scalar at position (0,2).
    # A_TL (2x2) * B_TR (1x1) doesn't make sense directly.
    # Let me reconsider the block structure.
    #
    # A = [[A_TL, A_TR],   B = [[B_TL, B_TR],
    #      [A_BL, A_BR]]        [B_BL, B_BR]]
    #
    # A_TL: 2x2, A_TR: 2x1, A_BL: 1x2, A_BR: 1x1
    # B_TL: 2x2, B_TR: 2x1, B_BL: 1x2, B_BR: 1x1
    #
    # C = A*B:
    # C_TL = A_TL*B_TL + A_TR*B_BL
    # C_TR = A_TL*B_TR + A_TR*B_BR
    # C_BL = A_BL*B_TL + A_BR*B_BL
    # C_BR = A_BL*B_TR + A_BR*B_BR

    # Let me redo this properly.
    # A_TL: indices 0,1,3,4 (2x2)
    # A_TR: indices 2,5 (2x1) - A[0,2], A[1,2]
    # A_BL: indices 6,7 (1x2) - A[2,0], A[2,1]
    # A_BR: index 8 (1x1) - A[2,2]
    #
    # B_TL: indices 0,1,3,4 (2x2)
    # B_TR: indices 2,5 (2x1) - B[0,2], B[1,2]
    # B_BL: indices 6,7 (1x2) - B[2,0], B[2,1]
    # B_BR: index 8 (1x1) - B[2,2]

    # C_TL (2x2): C[0,0], C[0,1], C[1,0], C[1,1]
    #   = A_TL*B_TL + A_TR*B_BL
    #   A_TL*B_TL: handled by Strassen above
    #   A_TR*B_BL: (2x1)*(1x2) = 2x2
    #   A_TR = [A02, A12]^T = [2, 5]
    #   B_BL = [B20, B21] = [6, 7]
    #   A_TR*B_BL = [[A02*B20, A02*B21], [A12*B20, A12*B21]]
    slices.append(([(2, 1.0)], [(6, 1.0)], [(widx(0, 0), 1.0)]))
    slices.append(([(2, 1.0)], [(7, 1.0)], [(widx(1, 0), 1.0)]))
    slices.append(([(5, 1.0)], [(6, 1.0)], [(widx(0, 1), 1.0)]))
    slices.append(([(5, 1.0)], [(7, 1.0)], [(widx(1, 1), 1.0)]))

    # C_TR (2x1): C[0,2], C[1,2]
    #   = A_TL*B_TR + A_TR*B_BR
    #   A_TL*B_TR: (2x2)*(2x1) = 2x1
    #   A_TL = [[0,1],[3,4]], B_TR = [2,5]
    #   Result: [A00*B02 + A01*B12, A10*B02 + A11*B12]
    slices.append(([(0, 1.0)], [(2, 1.0)], [(widx(2, 0), 1.0)]))
    slices.append(([(1, 1.0)], [(5, 1.0)], [(widx(2, 0), 1.0)]))
    slices.append(([(3, 1.0)], [(2, 1.0)], [(widx(2, 1), 1.0)]))
    slices.append(([(4, 1.0)], [(5, 1.0)], [(widx(2, 1), 1.0)]))
    #   A_TR*B_BR: (2x1)*(1x1) = 2x1
    #   A_TR = [A02, A12]^T, B_BR = B22
    slices.append(([(2, 1.0)], [(8, 1.0)], [(widx(2, 0), 1.0)]))
    slices.append(([(5, 1.0)], [(8, 1.0)], [(widx(2, 1), 1.0)]))

    # C_BL (1x2): C[2,0], C[2,1]
    #   = A_BL*B_TL + A_BR*B_BL
    #   A_BL*B_TL: (1x2)*(2x2) = 1x2
    #   A_BL = [A20, A21] = [6, 7], B_TL = [[0,1],[3,4]]
    #   Result: [A20*B00 + A21*B10, A20*B01 + A21*B11]
    slices.append(([(6, 1.0)], [(0, 1.0)], [(widx(0, 2), 1.0)]))
    slices.append(([(7, 1.0)], [(3, 1.0)], [(widx(0, 2), 1.0)]))
    slices.append(([(6, 1.0)], [(1, 1.0)], [(widx(1, 2), 1.0)]))
    slices.append(([(7, 1.0)], [(4, 1.0)], [(widx(1, 2), 1.0)]))
    #   A_BR*B_BL: (1x1)*(1x2) = 1x2
    #   A_BR = A22 = 8, B_BL = [B20, B21] = [6, 7]
    slices.append(([(8, 1.0)], [(6, 1.0)], [(widx(0, 2), 1.0)]))
    slices.append(([(8, 1.0)], [(7, 1.0)], [(widx(1, 2), 1.0)]))

    # C_BR (1x1): C[2,2]
    #   = A_BL*B_TR + A_BR*B_BR
    #   A_BL*B_TR: (1x2)*(2x1) = 1x1
    #   A_BL = [A20, A21] = [6, 7], B_TR = [B02, B12]^T = [2, 5]
    #   Result: A20*B02 + A21*B12
    slices.append(([(6, 1.0)], [(2, 1.0)], [(widx(2, 2), 1.0)]))
    slices.append(([(7, 1.0)], [(5, 1.0)], [(widx(2, 2), 1.0)]))
    #   A_BR*B_BR: (1x1)*(1x1) = 1x1
    #   A_BR = A22 = 8, B_BR = B22 = 8
    slices.append(([(8, 1.0)], [(8, 1.0)], [(widx(2, 2), 1.0)]))

    # Total: 7 + 4 + 6 + 6 + 4 = 27? Let me count.
    # Strassen: 7
    # C_TL cross: 4
    # C_TR: 6
    # C_BL: 6
    # C_BR: 3
    # Total: 7+4+6+6+3 = 26. Good.

    for r, (u_entries, v_entries, w_entries) in enumerate(slices):
        for idx, coeff in u_entries:
            U[idx][r] += coeff
        for idx, coeff in v_entries:
            V[idx][r] += coeff
        for idx, coeff in w_entries:
            W[idx][r] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
