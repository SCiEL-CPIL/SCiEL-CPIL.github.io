"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try to find a low-rank exact decomposition
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try block decomposition: 4=2+2, 5=3+2, 6=3+3
    # C = A*B where A is 4x5, B is 5x6
    # Split: A = [A_top; A_bot] (2x5, 2x5), B = [B_left | B_right] (5x3, 5x3)
    # C = [A_top; A_bot] * [B_left | B_right]
    # = [A_top*B_left, A_top*B_right; A_bot*B_left, A_bot*B_right]
    # Each block is 2x3 matrix product
    # 2x3x3 tensor has rank 18 (schoolbook: 2*3*3=18)
    # But we can do better?
    
    # For now, let's just return schoolbook if we can't find better quickly
    # The key insight: we need an EXACT decomposition with integer/half-integer entries
    
    # Try a simple rank reduction: can we combine some terms?
    # For 4x5x6, the tensor has 120 entries. 
    # A rank-r decomposition means we can write T as sum of r rank-1 tensors.
    
    # Let's try to use the fact that some linear combinations might cancel.
    # This is a hard problem, so for now return schoolbook.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
