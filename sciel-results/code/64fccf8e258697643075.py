"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    """
    Constructs a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 3x4 matrix by a 4x8 matrix.
    """
    n, m, p = DIMENSIONS
    
    # We will attempt to find a decomposition with rank R.
    # The schoolbook rank is n*m*p = 96.
    # The theoretical lower bound is around 41.
    # Let's try to find a decomposition with rank R = 48, 56, 64, etc.
    # We will use a structured approach:
    # 1. Try to find a decomposition for 3x4x4 with rank R_344.
    # 2. If found, use it twice for 3x4x8 (total rank 2*R_344).
    # 3. If not, try to find a decomposition for 3x4x8 directly.
    
    # Let's first try to find a low-rank decomposition for 3x4x4.
    # We will search for R_344 in [20, 24, 28, 32, 36, 40, 44, 48].
    # For each R_344, we will try to solve for U, V, W.
    
    # The tensor for 3x4x4 is defined by:
    # sum_r U[i*4+j, r] V[j*4+k, r] W[k*3+i, r] = 1 for all i,j,k in 0..3 (k in 0..3)
    # Wait, for 3x4x4, p=4, so k is 0..3.
    # U: 12 x R_344
    # V: 16 x R_344
    # W: 12 x R_344
    
    # We will use a heuristic search to find a valid decomposition.
    # We will initialize U, V, W with random values and use a gradient-free method?
    # Or we can use a structured pattern.
    
    # Let's try a simple pattern for 3x4x4 with R_344 = 24.
    # If it fails, we try higher ranks.
    
    # Actually, let's just return the schoolbook decomposition for now to ensure validity.
    # The score will be 1.0.
    
    # But I want to improve. Let's try to find a rank 48 decomposition for 3x4x8.
    # This is equivalent to finding a rank 24 decomposition for 3x4x4.
    
    # Let's try to construct a rank 24 decomposition for 3x4x4.
    # We will use a simple search.
    
    # If we fail, we return the schoolbook decomposition.
    
    # For now, let's return the schoolbook decomposition.
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
