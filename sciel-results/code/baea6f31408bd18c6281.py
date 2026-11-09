import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 135  # safety margin

    n, m, p = DIMENSIONS  # 4, 4, 7
    nm, mp, pn = n * m, m * p, p * n  # 16, 28, 28

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Strassen 2x2 definitions
    # Each term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (row, col, coeff) for A (2x2)
    # v_coeffs: list of (row, col, coeff) for B (2x2)
    # w_coeffs: list of (row, col, coeff) for C (2x2)

    strassen_2x2 = [
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3 = A00(B01-B11) -> C01 += P3, C11 += P3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4 = A11(B10-B00) -> C00 += P4, C10 += P4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    # Strategy: Use block decomposition
    # Split B into B_left (4x4) and B_right (4x3)
    # C = A @ B = A @ B_left + A @ B_right
    # 
    # For A @ B_left: use Strassen-of-Strassen (rank 49)
    # For A @ B_right: use block decomposition with Strassen for 2x2 blocks
    #   B_right is 4x3, split into 2x2 blocks:
    #   B_right = [[B00, B01], [B10, B11]] where each is 2x2 or 2x1
    #   Actually B_right is 4x3, so:
    #   B_right[0:2, 0:2] (2x2), B_right[0:2, 2] (2x1)
    #   B_right[2:4, 0:2] (2x2), B_right[2:4, 2] (2x1)
    #
    # Better approach: treat the whole thing as 4x4 x 4x7
    # Use the known rank 91 algorithm for 4x4 x 4x7
    # 
    # Actually, let's use a cleaner block decomposition:
    # 
    # A (4x4) = [[A00, A01], [A10, A11]]  (each 2x2)
    # B (4x7) = [[B00, B01], [B10, B11]]  where B00,B10 are 2x3, B01,B11 are 2x4
    # 
    # Wait, let me think more carefully.
    # 
    # The standard approach for 4x4 x 4x7:
    # Use the fact that 4x4 x 4x7 can be decomposed as:
    # - 4x4 x 4x4 (rank 49) + 4x4 x 4x3 (rank 42) = rank 91
    # 
    # For 4x4 x 4x3:
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]] where each Bi is 2x2 or 2x1
    # Actually B is 4x3, so:
    # B = [[B00, B01, B02], [B10, B11, B12]]
    # where B00,B10 are 2x1, B01,B11 are 2x1, B02,B12 are 2x1
    # 
    # Hmm, this is getting complicated. Let me use a different approach.
    #
    # Use the known algorithm: rank 91 for 4x4 x 4x7
    # This uses:
    # - 7 terms for the "main" part (like Strassen)
    # - Additional terms for the extra column
    #
    # Actually, the simplest valid approach:
    # Use schoolbook for the last column (4 terms) and Strassen-of-Strassen for the first 4 columns (49 terms)
    # Total rank: 49 + 4 = 53
    #
    # But we can do better. Let's use the block decomposition properly.
    #
    # For 4x4 x 4x3:
    # A (4x4) * B (4x3) = C (4x3)
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # Split B into 2x2 blocks: B = [[B00, B01], [B10, B11]] where B00,B10 are 2x2, B01,B11 are 2x1
    # Wait, B is 4x3, so:
    # B[0:2, 0:2] (2x2), B[0:2, 2] (2x1)
    # B[2:4, 0:2] (2x2), B[2:4, 2] (2x1)
    #
    # C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #      [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Each 2x2 x 2x2 product: rank 7 (Strassen)
    # Each 2x2 x 2x1 product: rank 4 (schoolbook, since 2*1=2, but we need 2 outputs, so 4 terms)
    #
    # Total for 4x4 x 4x3:
    # 4 * 7 (for 2x2 x 2x2) + 4 * 4 (for 2x2 x 2x1) = 28 + 16 = 44
    #
    # So total rank: 49 + 44 = 93
    #
    # Let's implement this.

    rank_444 = 49
    rank_443 = 44
    total_rank = rank_444 + rank_443

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # --- Part 1: 4x4 x 4x4 using Strassen-of-Strassen (rank 49) ---
    for s_idx, (X_def, Y_def, Z_def) in enumerate(strassen_2x2):
        for w_idx, (u_def, v_def, out_def) in enumerate(strassen_2x2):
            r = s_idx * 7 + w_idx

            # U column: coefficients for A (4x4)
            U_col = np.zeros(nm)
            for (bx, by, cx) in X_def:
                for (i_prime, j_prime, cu) in u_def:
                    i = 2 * bx + i_prime
                    j = 2 * by + j_prime
                    a_idx = i * m + j
                    U_col[a_idx] += cx * cu
            U[:, r] = U_col

            # V column: coefficients for B (4x4, cols 0-3)
            V_col = np.zeros(mp)
            for (bx, by, cy) in Y_def:
                for (i_prime, j_prime, cv) in v_def:
                    row_b = 2 * bx + i_prime
                    col_b = 2 * by + j_prime
                    b_idx = row_b * p + col_b
                    V_col[b_idx] += cy * cv
            V[:, r] = V_col

            # W column: coefficients for C (4x4, cols 0-3)
            W_col = np.zeros(pn)
            for (dx, dy, cd) in Z_def:
                for (i_prime, j_prime, co) in out_def:
                    row_c = 2 * dx + i_prime
                    col_c = 2 * dy + j_prime
                    c_idx = col_c * n + row_c
                    W_col[c_idx] += cd * co
            W[:, r] = W_col

    # --- Part 2: 4x4 x 4x3 (cols 4-6 of B) ---
    # Block decomposition:
    # A = [[A00, A01], [A10, A11]] (each 2x2)
    # B_right = [[B00, B01], [B10, B11]] where B00,B10 are 2x2 (cols 4,5), B01,B11 are 2x1 (col 6)
    # C_right = [[C00, C01], [C10, C11]] where C00,C10 are 2x2 (cols 4,5), C01,C11 are 2x1 (col 6)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # For each 2x2 x 2x2 product: 7 Strassen terms
    # For each 2x2 x 2x1 product: 4 schoolbook terms

    offset = rank_444
    col = 0

    # Define the 4 subproblems for the 2x2 x 2x2 parts
    # (A_block_row, A_block_col, B_row_block, C_col_offset)
    subproblems_222 = [
        (0, 0, 0, 4),  # A[0:2,0:2] * B[0:2,4:6] -> C[0:2,4:6]
        (0, 2, 2, 4),  # A[0:2,2:4] * B[2:4,4:6] -> C[0:2,4:6]
        (2, 0, 0, 4),  # A[2:4,0:2] * B[0:2,4:6] -> C[2:4,4:6]
        (2, 2, 2, 4),  # A[2:4,2:4] * B[2:4,4:6] -> C[2:4,4:6]
    ]

    for (ar, ac, br, cc) in subproblems_222:
        for (u_coeffs, v_coeffs, w_coeffs) in strassen_2x2:
            U_col = np.zeros(nm)
            V_col = np.zeros(mp)
            W_col = np.zeros(pn)

            # U: A[ar:ar+2, ac:ac+2]
            for (lr, lc, coeff) in u_coeffs:
                i = ar + lr
                j = ac + lc
                a_idx = i * m + j
                U_col[a_idx] += coeff

            # V: B[br:br+2, cc:cc+2]
            for (lr, lc, coeff) in v_coeffs:
                i = br + lr
                j = cc + lc
                b_idx = i * p + j
                V_col[b_idx] += coeff

            # W: C[ar:ar+2, cc:cc+2]
            for (lr, lc, coeff) in w_coeffs:
                i = ar + lr
                k = cc + lc
                c_idx = k * n + i
                W_col[c_idx] += coeff

            U[:, offset + col] = U_col
            V[:, offset + col] = V_col
            W[:, offset + col] = W_col
            col += 1

    # Now handle the 2x2 x 2x1 parts for column 6
    # C01 = A00*B01 + A01*B11
    # C11 = A10*B01 + A11*B11
    #
    # B01 = B[0:2, 6] (2x1)
    # B11 = B[2:4, 6] (2x1)
    #
    # C01[0] = A00[0,0]*B01[0] + A00[0,1]*B11[0] + A01[0,0]*B01[1] + A01[0,1]*B11[1]
    # C01[1] = A00[1,0]*B01[0] + A00[1,1]*B11[0] + A01[1,0]*B01[1] + A01[1,1]*B11[1]
    #
    # Wait, I need to be more careful. Let me use schoolbook for each 2x2 x 2x1.
    #
    # For A00 * B01:
    # Result[0] = A00[0,0]*B01[0] + A00[0,1]*B01[1]
    # Result[1] = A00[1,0]*B01[0] + A00[1,1]*B01[1]
    #
    # This requires 4 terms (2 for each output row).

    # Subproblems for 2x2 x 2x1
    subproblems_221 = [
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,6] -> C[0:2,6]
        (0, 2, 2, 1),  # A[0:2,2:4] * B[2:4,6] -> C[0:2,6]
        (2, 0, 0, 0),  # A[2:4,0:2] * B[0:2,6] -> C[2:4,6]
        (2, 2, 2, 1),  # A[2:4,2:4] * B[2:4,6] -> C[2:4,6]
    ]

    for (ar, ac, br, _) in subproblems_221:
        for lr in range(2):
            for j in range(2):
                i = ar + lr
                j_a = ac + j
                j_b = br + j
                k = 6

                a_idx = i * m + j_a
                b_idx = j_b * p + k
                c_idx = k * n + i

                U[a_idx, offset + col] = 1.0
                V[b_idx, offset + col] = 1.0
                W[c_idx, offset + col] = 1.0
                col += 1

    # Fast validation
    if time_left() > 5:
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        raise ValueError(f"Validation failed for target ({a},{b},{c}): {val}")

        # Check some non-target entries
        np.random.seed(42)
        for _ in range(200):
            if time_left() < 3:
                break
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            i_a, j_a = a // m, a % m
            j_b, k_b = b // p, b % p
            k_c, i_c = c // n, c % n
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > 1e-9:
                    raise ValueError(f"Validation failed for non-target ({a},{b},{c}): {val}")

    return {
        'U': U.tolist(),
        'V': V.tolist(),
        'W': W.tolist()
    }
