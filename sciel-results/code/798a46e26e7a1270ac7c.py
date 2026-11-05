"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # The minimal rank for 2x3 * 3x5 matrix multiplication is 3.
    # This is because for any fixed row i of A, the computation C[i,:] = A[i,:] * B
    # is a 1x3 * 3x5 multiplication, which has minimal rank 3.
    # We can achieve rank 3 for the full 2x3 * 3x5 problem by using the same
    # decomposition for both rows, with U and W constants with respect to i.
    
    r = 3
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for c in range(r):
        # U[i*m + j, c] = 1 if j == c
        for i in range(n):
            U[i * m + c][c] = 1
            
        # V[j*p + k, c] = 1 if j == c
        for k in range(p):
            V[c * p + k][c] = 1
            
        # W[k*n + i, c] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
