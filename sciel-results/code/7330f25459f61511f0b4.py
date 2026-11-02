"""
Matrix multiplication 3x5x7 rank minimization.

Approach: Schoolbook (naive) tensor decomposition.
The tensor M_{n,m,p} for multiplying an n x m matrix by an m x p matrix
has rank n*m*p in the standard basis. This program constructs that
decomposition exactly.

For n=3, m=5, p=7: rank = 105.

The output factor W follows the official trace convention W[k*n+i, r].
"""

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # U has shape (n*m, rank)
    # V has shape (m*p, rank)
    # W has shape (p*n, rank)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term in the sum: A[i,j] * B[j,k]
                # U row index: i*m + j
                # V row index: j*p + k
                # W row index: k*n + i  (official trace convention)
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
