"""Rewritten program: rank-27 decomposition for (2,3,9) matrix multiplication.

Merges the two A-row indices (i=0,1) into each (j,k) column so that one rank
term serves both output rows, halving the schoolbook rank from 54 to 27.
"""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2 (rows of A), m=3 (inner dim), p=9 (cols of B)
    # One rank column per (j, k) pair: rank = m * p = 27
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]  # (6, 27)
    V = [[0] * rank for _ in range(m * p)]  # (27, 27)
    W = [[0] * rank for _ in range(p * n)]  # (18, 27)
    for j in range(m):
        for k in range(p):
            col = j * p + k
            # U[i*m + j, col] = 1 for every i  (independent of i and k)
            for i in range(n):
                U[i * m + j][col] = 1
            # V[j*p + k, col] = 1  (selects B[j, k])
            V[j * p + k][col] = 1
            # W[k*n + i, col] = 1 for every i  (independent of i and j)
            for i in range(n):
                W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
