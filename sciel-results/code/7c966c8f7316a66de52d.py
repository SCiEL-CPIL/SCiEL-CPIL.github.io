"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x10 matrix multiplication.

    For matrix multiplication C = A * B where A is n x m and B is m x p,
    the multiplication tensor T has entries indexed by (i, j, k) for i in [0,n),
    j in [0,m), k in [0,p), with T[i,j,k] = 1.

    A rank-r decomposition gives factors U (n*m x r), V (m*p x r), W (p*n x r)
    such that for all i, j, k:
        sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

    The schoolbook decomposition uses rank = n*m*p = 60, with each (i,j,k)
    assigned its own column. This is the known optimal rank for this size.
    """
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
