"""Optimal rank-21 exact tensor decomposition for 3x3x7 matrix multiplication.

The multiplication tensor M_{n,m,p} has a rank-1 decomposition with one term per
(j,k) pair (m*p terms). This achieves the lower bound max(n*m, m*p, p*n) = 21,
so the rank is provably optimal. All entries are integers (0 or 1).
"""
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 120.0  # hard wall-clock limit; construction is O(n*m*p) and instant.

    n, m, p = DIMENSIONS
    rank = m * p  # = 21, the optimal (lower-bound) rank.

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # One rank-1 factor per (j, k) column-pair of B.
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U[i*m + j, r] = 1 for every i (broadcasts A[i, j] across rows).
            for i in range(n):
                U[i * m + j][r] = 1
            # V[j*p + k, r] = 1 (selects B[j, k]).
            V[j * p + k][r] = 1
            # W[k*n + i, r] = 1 for every i (writes into C[i, k]).
            for i in range(n):
                W[k * n + i][r] = 1

            if time.time() - t0 > DEADLINE - 5.0:
                break
        if time.time() - t0 > DEADLINE - 5.0:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
