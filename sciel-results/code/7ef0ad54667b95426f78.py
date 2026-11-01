"""Tensor decomposition for 2x2 by 2x4 matrix multiplication."""

# EVOLVE-BLOCK-START
def construct():
    # Matrix multiplication 2x2 by 2x4.
    # The tensor is a direct sum of 4 independent 2x2x1 multiplication tensors.
    # Rank of 2x2x1 is 4. Total rank = 4 * 4 = 16.
    n, m, p = 2, 2, 4
    rank = n * m * p
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
