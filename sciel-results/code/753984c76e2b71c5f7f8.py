"""Rank-30 decomposition of 2x3x10 matrix multiplication via shared middle-index terms."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10

    # Rank = m * p = 30
    # For each (j, k) pair, we have one term that contributes to both rows i=0 and i=1
    # U[i*m + j, (j,k)] = 1 for both i
    # V[j*p + k, (j,k)] = 1
    # W[k*n + i, (j,k)] = 1 for both i

    rank = m * p  # 30

    U = [[0] * rank for _ in range(n * m)]  # (6, 30)
    V = [[0] * rank for _ in range(m * p)]  # (30, 30)
    W = [[0] * rank for _ in range(p * n)]  # (20, 30)

    for j in range(m):
        for k in range(p):
            col = j * p + k
            # U factor: index by (i, j), set to 1 for both i values
            for i in range(n):
                U[i * m + j][col] = 1
            # V factor: index by (j, k)
            V[j * p + k][col] = 1
            # W factor: trace convention W[k*n + i, r], set to 1 for both i values
            for i in range(n):
                W[k * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
