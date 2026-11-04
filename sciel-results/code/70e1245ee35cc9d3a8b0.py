import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We reduce the problem to 2x3x3 and repeat for the 3 blocks of columns.
    # Rank of 2x3x3 schoolbook is 18.
    # We use rank 18 for the whole 2x3x9.
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for 2x3x3
    # i in 0..1, j in 0..2, c in 0..2
    col = 0
    for i in range(n):
        for j in range(m):
            for c in range(3):
                # U[i*m+j, col] = 1
                # V[j*3+c, col] = 1
                # W[c*n+i, col] = 1
                U[i * m + j][col] = 1
                # V and W are for the 2x3x3 block
                # We will replicate this pattern for all 3 blocks in V and W
                V[j * 3 + c][col] = 1
                W[c * n + i][col] = 1
                col += 1
                
    # Now replicate V and W for the other blocks
    # V is m x p = 3 x 9
    # V[j, k] where k = 3b + c
    # We set V[j, 3b+c, r] = V[j, c, r] (from the 2x3x3 solution)
    # But V in the loop above was set for k=c (0..2).
    # We need to set V[j, 3+c, r] and V[j, 6+c, r] to the same values.
    for j in range(m):
        for c in range(3):
            for r in range(rank):
                val = V[j * 3 + c][r]
                V[j * 3 + 3 + c][r] = val
                V[j * 3 + 6 + c][r] = val
                
    # W is p x n = 9 x 2
    # W[k, i] where k = 3b + c
    # We set W[3b+c, i, r] = W[c, i, r]
    for c in range(3):
        for i in range(n):
            for r in range(rank):
                val = W[c * n + i][r]
                W[(3 + c) * n + i][r] = val
                W[(6 + c) * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
