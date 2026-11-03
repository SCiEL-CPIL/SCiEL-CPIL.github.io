"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Direct construction of the schoolbook (rank n*m*p) tensor decomposition
    # for the 2x3 by 3x9 matrix multiplication, respecting the W[k*n+i, r]
    # output-trace convention. Each rank-1 term corresponds to a unique (i, j, k)
    # triple and sets U[i*m+j, c] = V[j*p+k, c] = W[k*n+i, c] = 1.
    n, m, p = 2, 3, 9
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
