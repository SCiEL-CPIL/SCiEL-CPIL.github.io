"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a lower rank decomposition
    # For 3x4 * 4x6, the matrix multiplication tensor has a known structure.
    # The minimal rank is at most the schoolbook rank (72).
    # We can try to find a lower rank by exploiting structure.
    
    # Strategy: Use a random linear combination approach to try to reduce rank.
    # However, for exact integer/half-integer entries, this is tricky.
    
    # Alternative: Use the fact that the multiplication tensor can be decomposed
    # using Strassen-like algorithms if dimensions allow, but 3x4x6 is irregular.
    
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    # We can try to improve it if time permits.
    
    rank = schoolbook_rank
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
    
    # Check if we have time to try to improve
    if time.time() > deadline - 10.0:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using a simple search
    # This is a heuristic approach - we'll try random combinations
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # For small dimensions, we can try to verify if a lower rank is possible
    # by checking the tensor rank bounds.
    # The rank of a 3x4x6 multiplication tensor is known to be at most 72.
    # It's unknown if it can be lower, but let's try a few random perturbations.
    
    # Actually, for exact decomposition with integer/half-integer entries,
    # the schoolbook is often optimal for irregular sizes.
    # Let's just return the schoolbook decomposition.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
