"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7-multiplication scheme for 2x2 matrices
    # M1 = (A00)(B10 - B00)
    # M2 = (A00 + A01)(B00)
    # M3 = (A11)(B01 - B11)
    # M4 = (A10)(B00 - B10)
    # M5 = (A10 + A11)(B11)
    # M6 = (A10 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B11 + B01)
    #
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # We have 3 column blocks of size 2, and 2 row blocks of size 2.
    # A is 2x4: rows split into A00 (row 0, cols 0-1) and A01 (row 1, cols 0-1)
    # Actually, let's think of A as having 2 row-blocks of height 1 and 2 col-blocks of width 2
    # No, better: A is 2x4. We can view it as a 2x2 block matrix where each block is 1x2.
    # That doesn't work for Strassen which needs square blocks.
    #
    # Alternative: view A as a 2x2 block matrix with blocks of size 1x2.
    # This doesn't directly fit Strassen. Let's reconsider.
    #
    # Actually, the standard approach for 2x4 * 4x6 is to note that we can write:
    # C = A * B where A is 2x4, B is 4x6, C is 2x6.
    # We can partition A into two 2x2 blocks side by side: A = [A_left, A_right]
    # And B into two 2x6 blocks top and bottom: B = [B_top; B_bottom]
    # Then C = A_left * B_top + A_right * B_bottom
    # Each product is 2x2 * 2x6 = 2x6.
    # For 2x2 * 2x6, we can use Strassen-like decomposition.
    #
    # For 2x2 * 2x6: let X be 2x2, Y be 2x6.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01, ..., y05], [y10, y11, ..., y15]]
    # Z = X*Y is 2x6:
    # Z[0, k] = x00*y0k + x01*y1k
    # Z[1, k] = x10*y0k + x11*y1k
    #
    # We can apply Strassen's idea column-wise. For each column k:
    # We need to compute x00*y0k + x01*y1k and x10*y0k + x11*y1k.
    # This is like multiplying two 2x1 matrices (column vectors) but with different X entries.
    #
    # Actually, a cleaner approach: use the 7-multiplication formula for 2x2*2x2 and apply
    # it separately for each pair of output columns, but that would be 3 * 7 = 21 per product,
    # and we have 2 products (left and right), giving 42.
    #
    # Let me define the 7 terms for a 2x2 * 2x2 multiplication:
    # Given X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]:
    # M1 = x00*(y10-y00)
    # M2 = (x00+x01)*y00
    # M3 = x11*(y01-y11)
    # M4 = x10*(y00-y10)
    # M5 = (x10+x11)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y11+y01)
    # Z00 = M1+M4+M5-M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M7

    # For 2x2 * 2x6, we have 3 column pairs. For each pair (k, k+1), we use the 7 terms.
    # So for X (2x2) * Y (2x6):
    # For col pair 0 (cols 0,1): use Y[:, 0:2]
    # For col pair 1 (cols 2,3): use Y[:, 2:4]
    # For col pair 2 (cols 4,5): use Y[:, 4:6]
    # Total: 3 * 7 = 21 terms.

    # For our problem: C = A_left * B_top + A_right * B_bottom
    # A_left is 2x2 (A[:, 0:2]), A_right is 2x2 (A[:, 2:4])
    # B_top is 2x6 (B[0:2, :]), B_bottom is 2x6 (B[2:4, :])
    # So total rank = 21 + 21 = 42.

    # Let's build the decomposition.
    # U has shape (n*m, rank) = (8, 42)
    # V has shape (m*p, rank) = (24, 42)
    # W has shape (p*n, rank) = (12, 42)

    # Index conventions:
    # U[i*m + j, c] where i in [0,2), j in [0,4)
    # V[j*p + k, c] where j in [0,4), k in [0,6)
    # W[k*n + i, c] where k in [0,6), i in [0,2)

    # The 7 terms for 2x2*2x2:
    # term 0 (M1): U gets x00, V gets (y10-y00), W gets contributions to Z00, Z11
    # term 1 (M2): U gets (x00+x01), V gets y00, W gets contributions to Z10, Z11
    # term 2 (M3): U gets x11, V gets (y01-y11), W gets contributions to Z01, Z11
    # term 3 (M4): U gets x10, V gets (y00-y10), W gets contributions to Z00, Z10
    # term 4 (M5): U gets (x10+x11), V gets y11, W gets contributions to Z00, Z01
    # term 5 (M6): U gets (x10-x00), V gets (y00+y10), W gets contribution to Z00
    # term 6 (M7): U gets (x01-x11), V gets (y11+y01), W gets contributions to Z00, Z11

    # Wait, let me re-check the Strassen formulas:
    # C00 = M1 + M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7

    # So:
    # M1 contributes to C00 (+1) and C11 (+1)
    # M2 contributes to C10 (+1) and C11 (-1)
    # M3 contributes to C01 (+1) and C11 (+1)
    # M4 contributes to C00 (+1) and C10 (+1)
    # M5 contributes to C00 (+1) and C01 (+1)
    # M6 contributes to C00 (+1)
    # M7 contributes to C00 (-1) and C11 (+1)

    def strassen_terms():
        """Return list of (U_coeff, V_coeff, W_coeff) for each of the 7 terms.
        U_coeff: dict mapping (i,j) to coefficient
        V_coeff: dict mapping (j,k) to coefficient
        W_coeff: dict mapping (k,i) to coefficient
        Here i in [0,2), j in [0,2) for the 2x2 X, and j in [0,2), k in [0,2) for 2x2 Y.
        """
        terms = []
        # M1 = x00 * (y10 - y00)
        terms.append((
            {(0, 0): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (1, 1): 1}
        ))
        # M2 = (x00 + x01) * y00
        terms.append((
            {(0, 0): 1, (0, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1}
        ))
        # M3 = x11 * (y01 - y11)
        terms.append((
            {(1, 1): 1},
            {(0, 1): 1, (1, 1): -1},
            {(0, 1): 1, (1, 1): 1}
        ))
        # M4 = x10 * (y00 - y10)
        terms.append((
            {(1, 0): 1},
            {(0, 0): 1, (1, 0): -1},
            {(0, 0): 1, (0, 1): 1}
        ))
        # M5 = (x10 + x11) * y11
        terms.append((
            {(1, 0): 1, (1, 1): 1},
            {(1, 1): 1},
            {(0, 0): 1, (1, 1): 1}
        ))
        # M6 = (x10 - x00) * (y00 + y10)
        terms.append((
            {(1, 0): 1, (0, 0): -1},
            {(0, 0): 1, (1, 0): 1},
            {(0, 0): 1}
        ))
        # M7 = (x01 - x11) * (y11 + y01)
        terms.append((
            {(0, 1): 1, (1, 1): -1},
            {(1, 1): 1, (0, 1): 1},
            {(0, 0): -1, (1, 1): 1}
        ))
        return terms

    terms = strassen_terms()

    # For A_left * B_top:
    # X = A_left = A[:, 0:2], so X[i, j] = A[i, j] for i in [0,2), j in [0,2)
    # Y = B_top = B[0:2, :], so Y[j, k] = B[j, k] for j in [0,2), k in [0,6)
    # Z = C[:, 0:6] but actually Z is 2x6, and we're computing it in 3 column pairs.
    # For col pair t (t=0,1,2), Y_t = Y[:, 2t:2t+2], so Y_t[j, s] = B[j, 2t+s] for s in [0,2)
    # Z_t[i, s] = C[i, 2t+s]

    # For A_right * B_bottom:
    # X = A_right = A[:, 2:4], so X[i, j] = A[i, 2+j] for i in [0,2), j in [0,2)
    # Y = B_bottom = B[2:4, :], so Y[j, k] = B[2+j, k] for j in [0,2), k in [0,6)
    # Z = C[:, 0:6], Z_t[i, s] = C[i, 2t+s]

    # Now build U, V, W.
    # First 21 columns: A_left * B_top
    # Next 21 columns: A_right * B_bottom

    for t in range(3):  # column pair
        for term_idx in range(7):  # Strassen term
            col = t * 7 + term_idx
            u_coeff, v_coeff, w_coeff = terms[term_idx]

            # For A_left * B_top:
            # X[i, j] = A[i, j] -> U[i*4 + j, col]
            for (xi, xj), coeff in u_coeff.items():
                U[xi * 4 + xj][col] += coeff

            # Y[j, k] = B[j, 2t + s] where s in [0,2)
            # v_coeff maps (yj, yk) to coeff, where yk in [0,2)
            for (yj, yk), coeff in v_coeff.items():
                V[yj * 6 + (2 * t + yk)][col] += coeff

            # W[k, i] = C[i, 2t + s] where s in [0,2)
            # w_coeff maps (zk, zi) to coeff, where zk in [0,2)
            for (zk, zi), coeff in w_coeff.items():
                W[(2 * t + zk) * 2 + zi][col] += coeff

    # For A_right * B_bottom: columns 21 to 41
    for t in range(3):  # column pair
        for term_idx in range(7):  # Strassen term
            col = 21 + t * 7 + term_idx
            u_coeff, v_coeff, w_coeff = terms[term_idx]

            # X[i, j] = A[i, 2+j] -> U[i*4 + (2+j), col]
            for (xi, xj), coeff in u_coeff.items():
                U[xi * 4 + (2 + xj)][col] += coeff

            # Y[j, k] = B[2+j, 2t + s]
            for (yj, yk), coeff in v_coeff.items():
                V[(2 + yj) * 6 + (2 * t + yk)][col] += coeff

            # W[k, i] = C[i, 2t + s]
            for (zk, zi), coeff in w_coeff.items():
                W[(2 * t + zk) * 2 + zi][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

