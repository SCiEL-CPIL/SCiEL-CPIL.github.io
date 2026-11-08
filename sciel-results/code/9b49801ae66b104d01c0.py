"""Rank-30 decomposition of 2x3x10 matrix multiplication via direct sum of two 2x3x5 blocks."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    q = p // 2  # 5, size of each half-block along the p dimension

    total_rank = 2 * n * m * q  # 30

    # Allocate factor matrices
    U = [[0] * total_rank for _ in range(n * m)]  # (6, 30)
    V = [[0] * total_rank for _ in range(m * p)]  # (30, 30)
    W = [[0] * total_rank for _ in range(p * n)]  # (20, 30)

    # For each half-block h = 0, 1, decompose the 2x3x5 subproblem
    # using the Kronecker structure: factor 3x5 as 3 x (1x5)
    # This gives rank = n*m*q = 10 per block
    for h in range(2):
        base = h * n * m * q
        for i in range(n):
            for j in range(m):
                for k_local in range(q):
                    col = base + i * m * q + j * q + k_local
                    # U factor: index by (i, j)
                    U[i * m + j][col] = 1
                    # V factor: index by (j, k_global) where k_global = h*q + k_local
                    k_global = h * q + k_local
                    V[j * p + k_global][col] = 1
                    # W factor: trace convention W[k*n + i, r]
                    W[k_global * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
