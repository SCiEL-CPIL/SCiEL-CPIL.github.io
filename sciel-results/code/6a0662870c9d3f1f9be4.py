"""Matrix multiplication 2x2 x 2x13 - tensor decomposition search."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for c in range(rank):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
