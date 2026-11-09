import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use standard Strassen-like or schoolbook?
    # Schoolbook rank is n*m*p = 216.
    # A known lower bound is roughly (2n)^3 = 512? No, Strassen is 2n^log2(7).
    # For n=6, 6^log2(7) is approx 6^2.8 = 26.
    # Actually, let's just use the schoolbook algorithm which is always correct.
    # Rank = 216.
    # This is a valid, exact decomposition.
    # U, V, W shapes: (36, 216), (36, 216), (36, 216)
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Map index r to (i, j, k)
    # r = i * m * p + j * p + k
    # U row: i * m + j
    # V row: j * p + k
    # W row: k * n + i
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                
                U[u_row][r] = 1
                V[v_row][r] = 1
                W[w_row][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
