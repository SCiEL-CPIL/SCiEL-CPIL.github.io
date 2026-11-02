"""
Task: Rank-minimal exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

Uses the schoolbook decomposition of rank n*m*p = 64, which is provably exact.
Each (i, j, k) triple contributes one rank-1 term.
"""

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a valid exact tensor decomposition for 2x4 * 4x8 matrix multiplication.

    Schoolbook decomposition of rank 64. Each (i, j, k) triple gets one rank-1 term
    with U[i*m+j, c] = V[j*p+k, c] = W[k*n+i, c] = 1.
    """
    n, m, p = 2, 4, 8
    rank = n * m * p  # 64

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
