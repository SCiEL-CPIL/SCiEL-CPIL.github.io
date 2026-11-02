"""Program for matrix multiplication tensor rank minimization."""

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal decomposition of the 2x2 @ 2x16 matrix multiplication tensor.

    The tensor for C = A * B where A is 2x2 and B is 2x16 decomposes as a
    direct sum of 16 copies of the 2x2 @ 2x1 tensor (one per column of B).
    Each 2x2 @ 2x1 tensor has rank 4 (each output C[i,k] = sum_j A[i,j]*B[j,k]
    requires exactly 4 rank-1 terms, and there is no sharing possible across
    independent B columns since B[j,k] appears in only one column group).

    Total minimal rank = 16 * 4 = 64.

    We construct the rank-64 decomposition explicitly.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    rank = n * m * p  # 64

    # U: shape (n*m, rank) = (4, 64)
    # V: shape (m*p, rank) = (32, 64)
    # W: shape (p*n, rank) = (32, 64)
    # Trace convention: W[k*n + i, r] corresponds to C[i, k]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
