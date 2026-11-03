"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # The best known rank for 5x5 matrix multiplication is 78.
    # However, implementing the full 78-rank decomposition is extremely complex.
    # 
    # Let's try to use a block decomposition approach that achieves a rank
    # lower than 125. We can split the 5x5 matrices into blocks and use
    # Strassen's algorithm for some of the blocks.
    #
    # Split into 2x2 blocks:
    # A = [A11(2x2) | A12(2x3)]
    #     [A21(3x2) | A22(3x3)]
    # B = [B11(2x2) | B12(2x3)]
    #     [B21(3x2) | B22(3x3)]
    # C = A*B
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # For the 2x2 blocks, we can use Strassen (rank 7 instead of 8).
    # For the 3x3 blocks, we can use a 3x3 decomposition.
    #
    # Actually, let's think about this differently. The tensor rank of matrix
    # multiplication <n,m,p> is the minimum number of rank-1 tensors needed
    # to represent it.
    #
    # For 5x5x5, the schoolbook rank is 125.
    # 
    # Let's try to use a known decomposition. One approach is to use the
    # fact that we can embed smaller matrix multiplications.
    #
    # Actually, let me just implement the schoolbook decomposition but try
    # to find a better one through optimization.
    #
    # For now, let's return the schoolbook decomposition with rank 125.
    # This is the seed value.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
