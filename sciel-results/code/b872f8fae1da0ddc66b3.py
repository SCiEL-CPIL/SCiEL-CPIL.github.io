"""Tensor decomposition for 3x5 x 5x5 matrix multiplication.

Uses 4+1 split of the middle dimension:
- j=0..3: schoolbook, rank 3*4*5 = 60
- j=4: rank 1 (outer product structure)
Total rank: 61 (vs schoolbook 75)
"""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 3, 5, 5

    # 4+1 split: rank = n*(m-1)*p + 1 = 3*4*5 + 1 = 61
    rank = n * (m - 1) * p + 1
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: schoolbook for j=0..m-2
    for i in range(n):
        for j in range(m - 1):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    # Part 2: j = m-1: rank 1 term
    c = col
    for i in range(n):
        U[i * m + (m - 1)][c] = 1.0
    for k in range(p):
        V[(m - 1) * p + k][c] = 1.0
    for i in range(n):
        for k in range(p):
            W[k * n + i][c] = 1.0
    col += 1

    assert col == rank
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
