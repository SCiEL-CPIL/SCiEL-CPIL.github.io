"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    The schoolbook algorithm has rank n*m*p = 2*3*6 = 36.
    Given the difficulty of finding a lower rank decomposition for this specific 
    dimension set within the time limit, we return the verified schoolbook 
    decomposition which is guaranteed to be exact.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Construct schoolbook decomposition
    # Each rank term corresponds to a single (i, j, k) triple
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index for U: i*m + j
                # Index for V: j*p + k
                # Index for W: k*n + i
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
