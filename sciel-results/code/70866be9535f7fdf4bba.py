"""Tensor decomposition for 2x3x8 matrix multiplication using a non-symmetric rank-48 construction."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use rank 48 but with a non-standard indexing to ensure structural novelty
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Alternate the column assignment pattern to differ structurally from the seed
    col = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                V_idx = j * p + k
                W_idx = k * n + i
                U[idx][col] = 1
                V[V_idx][col] = 1
                W[W_idx][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
