"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    rank0 = n * m * p
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Try to find a lower-rank decomposition
    # Strategy: use block decomposition
    # Split B (4x7) into B1 (4x4) and B2 (4x3)
    # C = A*B1 + A*B2
    # For A*B1 (4x4 x 4x4): use a lower-rank decomposition
    # For A*B2 (4x4 x 4x3): use schoolbook or lower
    
    # Let's try a Strassen-like approach for the 4x4 x 4x4 part
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] each 2x2
    # B1 = [[B11, B12], [B21, B22]] each 2x2
    # C1 = [[C11, C12], [C21, C22]] each 2x2
    #
    # Strassen for 2x2:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # This uses 7 multiplications instead of 8.
    # But we need to track the actual tensor decomposition.
    
    # For now, let's just return the schoolbook to ensure correctness.
    # A proper implementation would search for lower-rank decompositions.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
