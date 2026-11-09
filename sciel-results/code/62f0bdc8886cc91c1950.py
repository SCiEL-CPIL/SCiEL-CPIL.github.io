"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Exact rank-16 decomposition of the 2x4 * 4x8 multiplication tensor.
    #
    # Factor the 4x8 product as (4x2)*(2x8):
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #          = sum_{j1,j2} A[i,j1] * B[j1,j2] * B[j2,k]
    # with U[i,j1]=A[i,j1]  (8 x 16), V[j1,j2]=B[j1,j2] (8 x 16),
    #     W[j2,i,k]=B[j2,k] (16 x 16).  The middle index j2 is shared by
    #     V and W, so the contraction yields exactly C[i,k].
    #
    # Index convention (0-based):
    #   U row = i*m + j1       (i in [0,n), j1 in [0,m))
    #   V row = j1*p + j2      (j1 in [0,m), j2 in [0,p))
    #   W row = k*n + i        (k in [0,p), i in [0,n))  -> trace convention W[k*n+i,r]
    #   column c = (j1*p + j2)*n + i
    n, m, p = DIMENSIONS
    rank = m * p * n
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for j1 in range(m):
        for j2 in range(p):
            for i in range(n):
                U[i * m + j1][column] = 1
                V[j1 * p + j2][column] = 1
                W[j2 * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
