import numpy as np
import time

DIMENSIONS = (4, 4, 7)

def construct():
    start_time = time.time()
    DEADLINE = 140

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

    rank_444 = 49
    rank_443 = 44
    total_rank = rank_444 + rank_443

    U = np.zeros((nm, total_rank), dtype=float)
    V = np.zeros((mp, total_rank), dtype=float)
    W = np.zeros((pn, total_rank), dtype=float)

    # --- Part 1: 4x4 x 4x4 using Strassen-of-Strassen (rank 49) ---
    # Outer Strassen: 7 terms, each decomposing 2x2 block mult
    # Inner Strassen: 7 terms, each decomposing scalar mult (trivial)

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
    # 4 subproblems: (2x2) x (2x3), each rank 11
    # For each subproblem: 7 Strassen terms for first 2 cols + 4 schoolbook for 3rd col

    offset = rank_444
    col = 0

    subproblems = [
        (0, 0, 0, 0),  # A[0:2,0:2] * B[0:2,4:7] -> C[0:2,4:7]
        (0, 2, 2, 0),  # A[0:2,2:4] * B[2:4,4:7] -> C[0:2,4:7]
        (2, 0, 0, 2),  # A[2:4,0:2] * B[0:2,4:7] -> C[2:4,4:7]
        (2, 2, 2, 2),  # A[2:4,2:4] * B[2:4,4:7] -> C[2:4,4:7]
    ]

    for (ar, ac, br, cr) in subproblems:
        # Strassen terms for first 2 columns (global cols 4,5)
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

            # V: B[br:br+2, 4:6] (first 2 of the 3 cols)
            for (lr, lc, coeff) in v_coeffs:
                i = br + lr
                j = 4 + lc
                b_idx = i * p + j
                V_col[b_idx] += coeff

            # W: C[cr:cr+2, 4:6]
            for (lr, lc, coeff) in w_coeffs:
                i = cr + lr
                k = 4 + lc
                c_idx = k * n + i
                W_col[c_idx] += coeff

            U[:, offset + col] = U_col
            V[:, offset + col] = V_col
            W[:, offset + col] = W_col
            col += 1

        # Schoolbook for 3rd column (global col 6)
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

    # Fast validation using tensor contraction check
    if time_left() > 5:
        # Check all target entries efficiently
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
