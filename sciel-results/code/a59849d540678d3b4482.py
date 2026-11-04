"""
Winograd-style decomposition for 4x5 by 5x6 matrix multiplication.
"""
import numpy as np

DIMENSIONS = [4, 5, 6]

# Winograd 4-point FFT matrices (using complex roots of unity for correctness)
# For real Winograd, we use the standard 4-point DFT/IDFT matrices
# W4: 4-point DFT matrix (forward)
# W4_H: 4-point IDFT matrix (with 1/4 scaling)

def construct():
    n, m, p = 4, 5, 6
    
    # Schoolbook decomposition: rank = n*m*p = 120
    # U shape: (n*m, rank) = (20, 120)
    # V shape: (m*p, rank) = (30, 120)  
    # W shape: (p*n, rank) = (24, 120)
    
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
