"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook (element-wise) tensor decomposition for multiplying
    a 2x3 matrix by a 3x8 matrix.

    The decomposition has rank n*m*p = 2*3*8 = 48.
    For each (i, j, k) triple we have one rank-1 term:
        U[i*m + j, c] = 1
        V[j*p + k, c] = 1
        W[k*n + i, c] = 1
    where c is the column index (0..rank-1).

    This yields: (A*B)[i,k] = sum_{j} A[i,j]*B[j,k]
    which is recovered as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
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
