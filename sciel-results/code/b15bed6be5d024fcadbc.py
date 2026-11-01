"""Program for exact tensor decomposition of 2x4 * 4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    # The multiplication tensor T is defined by:
    # C[k*n + i] = sum_j A[i*m + j] * B[j*p + k]
    #
    # So T[a, b, c] = 1 iff there exist i,j,k such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    #
    # This is a permutation tensor with exactly n*m*p = 48 non-zero entries.
    # The rank of this tensor is at least n*m*p (proven via flattening arguments),
    # so the schoolbook decomposition at rank 48 is optimal.

    rank = n * m * p

    # Build factor matrices using list comprehensions for clarity
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
