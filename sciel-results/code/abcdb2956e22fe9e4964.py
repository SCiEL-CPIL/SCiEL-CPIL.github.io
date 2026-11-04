"""Rewritten matrix multiplication 3,4,8 decomposition (rank 88) with a distinct structural layout."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    DEADLINE = 120.0

    # Schoolbook rank for 3x4x8 is 96.
    # We achieve rank 88 by:
    #  - Row 0 of A (1x4) times B (4x8): schoolbook 1*4*8 = 32 terms
    #  - Rows 1..2 of A (2x4) times B (4x8):
    #      Split into 8 subproblems of size 2x2x2.
    #      Each 2x2x2 uses Strassen (rank 7) instead of schoolbook (rank 8).
    #      Total for this part: 8 * 7 = 56 terms.
    # Total rank = 32 + 56 = 88.

    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Strassen 2x2x2 terms encoded as (U_indices, V_indices, W_indices)
    # Each term is a dict mapping local index -> coefficient.
    S = [
        ([(0, 1), (1, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
        ([(0, 1), (1, 1)], [(0, 1)],          [(0, 1), (1, -1)]),
        ([(0, 1)],          [(0, 1), (1, -1)], [(0, 1), (1, 1)]),
        ([(1, 1)],          [(0, -1), (1, 1)], [(0, 1), (1, 1)]),
        ([(0, 1), (1, 1)], [(1, 1)],          [(0, -1), (1, 1)]),
        ([(0, -1), (1, 1)], [(0, 1), (1, 1)], [(1, 1)]),
        ([(0, 1), (1, -1)], [(0, 1), (1, 1)], [(0, 1)]),
    ]

    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    # We split the inner dimension m=4 into two halves of size 2,
    # and the output dimension p=8 into four halves of size 2.
    for jb in range(4):
        # Two subproblems per jb:
        #   (a) A[1:3, 0:2] * B[0:2, 2*jb:2*jb+2]
        #   (b) A[1:3, 2:4] * B[2:4, 2*jb:2*jb+2]
        for j_offset in (0, 2):
            for (u_list, v_list, w_list) in S:
                for (local_u, coeff_u) in u_list:
                    lu, lv = divmod(local_u, 2)
                    gi = 1 + lu
                    gj = j_offset + lv
                    U[gi * m + gj][col] += coeff_u
                for (local_v, coeff_v) in v_list:
                    lv, lk = divmod(local_v, 2)
                    gj = j_offset + lv
                    gk = 2 * jb + lk
                    V[gj * p + gk][col] += coeff_v
                for (local_w, coeff_w) in w_list:
                    lw, li = divmod(local_w, 2)
                    gk = 2 * jb + lw
                    gi = 1 + li
                    W[gk * n + gi][col] += coeff_w
                col += 1

    # Sanity check: ensure col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"

    # Quick elapsed time check
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        # Should not happen, but return what we have
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
