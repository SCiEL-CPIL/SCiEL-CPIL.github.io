"""
Task: Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

Uses the canonical schoolbook decomposition of rank n*m*p = 64, which is provably exact.
Each rank-1 term corresponds to a unique (i, j, k) triple.
"""

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for idx, (i, j, k) in enumerate(
        (i, j, k) for i in range(n) for j in range(m) for k in range(p)
    ):
        U[i * m + j][idx] = 1.0
        V[j * p + k][idx] = 1.0
        W[k * n + i][idx] = 1.0
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
