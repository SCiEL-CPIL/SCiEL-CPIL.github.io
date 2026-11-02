"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Use a block decomposition approach to reduce rank below schoolbook 125.
    # Split 5x5 into 3x3 + 2x2 blocks.
    # A = [A11(3x3) | A12(3x2)]
    #     [A21(2x3) | A22(2x2)]
    # B = [B11(3x3) | B12(3x2)]
    #     [B21(2x3) | B22(2x2)]
    # C = A*B
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Use Strassen for 3x3 (rank 7) and direct multiplication for others.
    # But this doesn't directly give a tensor decomposition with lower rank.
    # 
    # Instead, let's use a simpler approach: implement the schoolbook decomposition
    # but with a more efficient structure.
    # 
    # Actually, the best known rank for 5x5 is 78, but implementing that is complex.
    # Let's try a block decomposition that achieves rank 107 or better.
    
    # For a 5x5 matrix multiplication, we can use the following block structure:
    # Split into 2+3 blocks:
    # A = [A11(2x2) | A12(2x3)]
    #     [A21(3x2) | A22(3x3)]
    # B = [B11(2x2) | B12(2x3)]
    #     [B21(3x2) | B22(3x3)]
    # C = A*B
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Use Strassen for 2x2 (rank 7) and direct for others.
    # But again, this doesn't directly reduce the tensor rank.
    
    # Let's just implement the schoolbook decomposition for now, as it's the safest.
    # The rank is 125, which is the seed value.
    
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
