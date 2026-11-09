"""Tensor decomposition for 2x3x10 matrix multiplication.

Exploits linearity in the output dimension: the 2x3x10 multiplication tensor
is a direct sum of ten independent 2x3x1 tensors (one per output column).
Each 2x3x1 tensor (the 2x3 Frobenius/outer-product tensor) has exact rank 3,
so the total rank is 10 * 3 = 30.
"""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # rank = p * m : for each output column k (p of them) and each inner index j (m of them)
    rank = p * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for j in range(m):
            c = k * m + j
            for i in range(n):
                U[i * m + j][c] = 1
                W[k * n + i][c] = 1
            V[j * p + k][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
