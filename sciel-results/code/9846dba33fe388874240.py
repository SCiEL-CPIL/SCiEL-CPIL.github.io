"""Block-wise Kronecker decomposition for 3x5x6 matrix multiplication.

Treats the 3x5 matrix as a 1x3 block row of 1x5 vectors and the 5x6 matrix
as a 5x1 block column of 1x6 vectors. The base 1x5 by 5x6 tensor has rank 5
(diagonal terms), and tensoring with the 3x1x1 identity (rank 3) yields
rank 3*5 = 15.
"""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 3 * 5 = 15

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j_base in range(m):
            t = i * m + j_base
            for k in range(p):
                U[i * m + j_base][t] = 1
                V[j_base * p + k][t] = 1
                W[k * n + i][t] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
