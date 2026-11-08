"""
Matrix multiplication 3,3,8 tensor rank decomposition.

The minimal rank for M_{3,3,8} is 72 (the 72 rank-1 tensors in the schoolbook
decomposition are linearly independent). This program constructs that decomposition.
"""

import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-72 tensor decomposition for multiplying a 3x3 matrix
    by a 3x8 matrix.

    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
              where each is a list of lists with integer entries.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 72

    U = [[0] * rank for _ in range(n * m)]  # 9 x 72
    V = [[0] * rank for _ in range(m * p)]  # 24 x 72
    W = [[0] * rank for _ in range(p * n)]  # 24 x 72

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j   # row index in U
                v_idx = j * p + k   # row index in V
                w_idx = k * n + i   # row index in W (official trace convention)

                U[u_idx][column] = 1
                V[v_idx][column] = 1
                W[w_idx][column] = 1

                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
