"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

# Known optimal rank for 3x5 * 5x7 is 105 (n*m*p) is schoolbook.
# Actually, for small matrices, the rank can be lower.
# Let's try to find a lower rank decomposition.
# The rank of matrix multiplication tensor for n x m and m x p is at most n*m*p.
# Lower bounds: max(n*p, m*p, n*m) is not the rank bound, but the tensor rank is >= max(n*p, m*p, n*m) is false.
# The tensor rank is >= n*m*p / max(n,m,p)? No.
# The rank is at least max(n*p, m*p, n*m) is also not true.
# The rank is at least n*m*p / max(n,m,p) is a known lower bound? No.
# Actually, the rank of the matrix multiplication tensor is known to be n*m*p for small cases?
# For 2x2 * 2x2, rank is 7 (Strassen).
# For 3x3 * 3x3, rank is 23.
# For 3x5 * 5x7, we can try to find a decomposition.

# Let's implement a simple search or use a known result.
# Since I don't have a known result for 3x5 * 5x7, I will try to construct one.
# The schoolbook rank is 3*5*7 = 105.
# Can we do better?
# Let's try to use a greedy algorithm or random initialization with refinement.
# But finding an exact tensor decomposition is hard.
# For now, I will return the schoolbook decomposition, which is guaranteed to work.

def construct():
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
