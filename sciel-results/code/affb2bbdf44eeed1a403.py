"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Construct rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen terms for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[x, y], [z, w]]
    # C = [[u, v], [s, t]]
    # M1 = (a+d)(x+w)
    # M2 = (c+d)x
    # M3 = a(w-z)
    # M4 = d(y-x)
    # M5 = (a+b)z
    # M6 = (c+a)(y-w)
    # M7 = (b+d)(x-z)
    
    # Mapping to indices:
    # U indices: 0:00, 1:01, 2:10, 3:11
    # V indices: 0:00, 1:01, 2:10, 3:11 (for 2x2 block)
    # W indices: 0:00, 1:01, 2:10, 3:11 (for 2x2 block)
    
    # Block k (0..6) corresponds to columns 2k, 2k+1 of B
    # V row for (j, col) is j*p + col
    # W row for (col, i) is col*n + i
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7

        # M1 = (a+d)(x+w) -> U[00], U[11] * V[00], V[11] -> W[00], W[11]
        U[0][base_col + 0] = 1
        U[3][base_col + 0] = 1
        V[k1][base_col + 0] = 1
        V[p + k2][base_col + 0] = 1
        W[k1 * n + 0][base_col + 0] = 1
        W[k2 * n + 1][base_col + 0] = 1

        # M2 = (c+d)x -> U[10], U[11] * V[00] -> W[10], W[11]
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1

        # M3 = a(w-z) -> U[00] * V[11], V[10] -> W[01], W[11]
        U[0][base_col + 2] = 1
        V[p + k2][base_col + 2] = 1
        V[p + k1][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1

        # M4 = d(y-x) -> U[11] * V[01], V[00] -> W[00], W[01]
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1

        # M5 = (a+b)z -> U[00], U[01] * V[10] -> W[00], W[10]
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        V[p + k1][base_col + 4] = 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1

        # M6 = (c+a)(y-w) -> U[10], U[00] * V[01], V[11] -> W[01], W[11]
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        V[k2][base_col + 5] = 1
        V[p + k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = 1

        # M7 = (b+d)(x-z) -> U[01], U[11] * V[00], V[10] -> W[00], W[10]
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
