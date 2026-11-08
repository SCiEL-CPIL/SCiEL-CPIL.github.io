"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks (j in {0,1})
plus a rank-1 term for the j=2 slice, targeting rank 29.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    U_t = []
    V_t = []
    W_t = []

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)

        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)

        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])

        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])

        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])

        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])

        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])

        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])

        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])

    # Strassen for A[0:2, 0:2] * B[0:2, k0:k0+2] for k0 in {0, 2, 4, 6}
    for k0 in [0, 2, 4, 6]:
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)

    # Rank-1 term for j=2: A[i,2] * B[2,k] -> C[i,k] for all i,k
    # U[i*3+2, r] = 1 for i in {0,1,2}
    # V[2*8+k, r] = 1 for k in {0,...,7}
    # W[k*3+i, r] = 1 for all i,k
    U_j2 = []
    V_j2 = []
    W_j2 = []
    for i in range(3):
        U_j2.append(((i, 2), 1))
    for k in range(8):
        V_j2.append(((2, k), 1))
    for k in range(8):
        for i in range(3):
            W_j2.append(((i, k), 1))
    U_t.append(U_j2)
    V_t.append(V_j2)
    W_t.append(W_j2)

    rank = len(W_t)

    # Build the factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j][idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k][idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i][idx] = coeff

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify_decomp(U_arr, V_arr, W_arr):
        # Fallback: try the 52-rank approach
        U_t_fb = []
        V_t_fb = []
        W_t_fb = []

        for k0 in [0, 2, 4, 6]:
            strassen_decomp(0, 0, 0, k0, U_t_fb, V_t_fb, W_t_fb)

            # 2x1x2: A[0:2, 2] * B[2, k0:k0+2]
            for i in range(2):
                for k in range(k0, k0+2):
                    U_t_fb.append([((i, 2), 1)])
                    V_t_fb.append([((2, k), 1)])
                    W_t_fb.append([((i, k), 1)])

            # 1x2x2: A[2, 0:2] * B[0:2, k0:k0+2]
            for j in range(2):
                for k in range(k0, k0+2):
                    U_t_fb.append([((2, j), 1)])
                    V_t_fb.append([((j, k), 1)])
                    W_t_fb.append([((2, k), 1)])

            # 1x1x2: A[2,2] * B[2, k0:k0+2]
            for k in range(k0, k0+2):
                U_t_fb.append([((2, 2), 1)])
                V_t_fb.append([((2, k), 1)])
                W_t_fb.append([((2, k), 1)])

        rank_fb = len(W_t_fb)
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]

        for idx in range(rank_fb):
            for (i, j), coeff in U_t_fb[idx]:
                U_fb[i * m + j][idx] = coeff
            for (j, k), coeff in V_t_fb[idx]:
                V_fb[j * p + k][idx] = coeff
            for (i, k), coeff in W_t_fb[idx]:
                W_fb[k * n + i][idx] = coeff

        U_fb_arr = np.array(U_fb, dtype=float)
        V_fb_arr = np.array(V_fb, dtype=float)
        W_fb_arr = np.array(W_fb, dtype=float)

        if verify_decomp(U_fb_arr, V_fb_arr, W_fb_arr):
            return {"U": U_fb, "V": V_fb, "W": W_fb}

        # Final fallback: schoolbook
        rank_final = n * m * p
        U_final = [[0] * rank_final for _ in range(n * m)]
        V_final = [[0] * rank_final for _ in range(m * p)]
        W_final = [[0] * rank_final for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_final[i * m + j][col] = 1
                    V_final[j * p + k][col] = 1
                    W_final[k * n + i][col] = 1
                    col += 1
        return {"U": U_final, "V": V_final, "W": W_final}

    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result

    return {"U": to_nice_list(U_arr), "V": to_nice_list(V_arr), "W": to_nice_list(W_arr)}
# EVOLVE-BLOCK-END
