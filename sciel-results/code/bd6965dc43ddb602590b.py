import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 32 decomposition
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Split B into two 4x4 blocks
    # Block 1: cols 0-3 of B, output cols 0-3 of C
    for i in range(n):
        for j in range(m):
            for k in range(4): # k is 0..3
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: cols 4-7 of B, output cols 4-7 of C
    for i in range(n):
        for j in range(m):
            for k in range(4): # k is 0..3, but actual col is 4+k
                actual_k = 4 + k
                U[i * m + j][col] = 1
                V[j * p + actual_k][col] = 1
                W[k * n + i][col] = 1 # Output col index is k (0..3) for the second half?
                # Wait. C is 2x8.
                # C[i, 0..3] comes from B_1.
                # C[i, 4..7] comes from B_2.
                # W index is k_out * n + i.
                # For B_2, the output column in C is 4+k.
                # So W index should be (4+k) * n + i.
                W[actual_k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
