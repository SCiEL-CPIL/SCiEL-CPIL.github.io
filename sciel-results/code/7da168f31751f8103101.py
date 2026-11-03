"""Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix."""

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    schoolbook_rank = n * m * p  # 64

    # Start with schoolbook decomposition
    U = [[0] * schoolbook_rank for _ in range(nm)]
    V = [[0] * schoolbook_rank for _ in range(mp)]
    W = [[0] * schoolbook_rank for _ in range(pn)]
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
