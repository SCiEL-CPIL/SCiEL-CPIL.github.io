"""Program for rank-minimal exact tensor decomposition of M_{2,3,3}."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = n*m*p = 18
    # We will build the standard decomposition and return it.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U indexed by (i, j) -> row i*m + j
                # V indexed by (j, k) -> row j*p + k
                # W indexed by (k, i) -> row k*n + i  (trace convention W[k*n+i, r])
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
