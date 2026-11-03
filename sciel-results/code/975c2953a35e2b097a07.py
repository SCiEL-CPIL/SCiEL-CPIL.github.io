import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8

    # Rank-56 decomposition via 8 independent 2x2 Strassen blocks.
    # Each block handles a 2x2 subproblem. We tile the 2x4 @ 4x8 product
    # into 8 blocks of size (2x2) @ (2x2) -> (2x2).
    #
    # The tensor T has dimensions:
    #   U: n*m = 8 rows
    #   V: m*p = 32 rows
    #   W: p*n = 16 rows
    #
    # For each block (jj, half) where jj in [0,4) and half in [0,2):
    #   A sub-block: rows 0..1, cols 2*half..2*half+1  (2x2)
    #   B sub-block: rows 2*half..2*half+1, cols 2*jj..2*jj+1  (2x2)
    #   C sub-block: rows 0..1, cols 2*jj..2*jj+1  (2x2)
    #
    # Strassen rank-7 local factors (U_loc, V_loc, W_loc each 4x7):
    U_loc = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    V_loc = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    W_loc = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    total_rank = 8 * 7  # 56
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    for jj in range(p // 2):
        for half in range(2):
            base_r = (jj * 2 + half) * 7

            # A indices: i*m + j for i in {0,1}, j in {2*half, 2*half+1}
            a_idx = [0 * m + (2 * half + 0),
                     0 * m + (2 * half + 1),
                     1 * m + (2 * half + 0),
                     1 * m + (2 * half + 1)]
            # B indices: j*p + k for j in {2*half, 2*half+1}, k in {2*jj, 2*jj+1}
            b_idx = [(2 * half + 0) * p + (2 * jj + 0),
                     (2 * half + 0) * p + (2 * jj + 1),
                     (2 * half + 1) * p + (2 * jj + 0),
                     (2 * half + 1) * p + (2 * jj + 1)]
            # W indices: k*n + i for k in {2*jj, 2*jj+1}, i in {0,1}
            c_idx = [(2 * jj + 0) * n + 0,
                     (2 * jj + 1) * n + 0,
                     (2 * jj + 0) * n + 1,
                     (2 * jj + 1) * n + 1]

            for r in range(7):
                rr = base_r + r
                U[a_idx[0]][rr] += U_loc[0][r]
                U[a_idx[1]][rr] += U_loc[1][r]
                U[a_idx[2]][rr] += U_loc[2][r]
                U[a_idx[3]][rr] += U_loc[3][r]

                V[b_idx[0]][rr] += V_loc[0][r]
                V[b_idx[1]][rr] += V_loc[1][r]
                V[b_idx[2]][rr] += V_loc[2][r]
                V[b_idx[3]][rr] += V_loc[3][r]

                W[c_idx[0]][rr] += W_loc[0][r]
                W[c_idx[1]][rr] += W_loc[1][r]
                W[c_idx[2]][rr] += W_loc[2][r]
                W[c_idx[3]][rr] += W_loc[3][r]

    if time.time() - start_time > 110:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
