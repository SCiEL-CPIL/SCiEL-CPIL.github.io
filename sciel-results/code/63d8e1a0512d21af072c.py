"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 8 decomposition using row-wise splitting
    # Each row of A (size 1xm) multiplied by B (mxp) gives a row of C (1xp)
    # Rank of 1xm x mxp is m
    # Total rank is n * m
    rank = n * m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each row i of A (and C), we use m terms
    # Term r corresponds to column j of A (and row of B)
    # U[i*m + j, r] = 1 if r == i*m + j
    # V[j*p + k, r] = 1 if r // m == i and r % m == j ? No.
    # Let's map r to (i, j).
    # r = i * m + j.
    # U index: i * m + j.
    # U[i*m + j, r] = 1 if r == i*m + j.
    # V index: j * p + k.
    # V[j*p + k, r] = 1 if r // m == i and r % m == j.
    # Wait, V depends on j. If r corresponds to (i, j), then V should be 1 for that j.
    # W index: k * n + i.
    # W[k*n + i, r] = 1 if r // m == i.
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U
            U[i * m + j][r] = 1
            # V
            for k in range(p):
                V[j * p + k][r] = 1
            # W
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
