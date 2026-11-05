"""Matrix multiplication 3,3,8 — schoolbook decomposition (rank 72)."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p:
    #   U has shape (n*m, rank), indexed by (i*m + j)
    #   V has shape (m*p, rank), indexed by (j*p + k)
    #   W has shape (p*n, rank), indexed by (k*n + i)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
