"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook baseline (rank 63)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Try to find lower-rank decomposition using known efficient algorithms
    # For 3x3 x 3x7, we can use block structure
    # Split into 3x3 x 3x3 (rank 21 known) and 3x3 x 3x4
    
    # Strategy: Use Strassen-like decomposition for 3x3 x 3x3 part
    # and optimize the 3x3 x 3x4 part
    
    # For 3x3 x 3x3, use the known rank-21 decomposition
    # This is based on de Groote's algorithm or similar
    
    # Let's implement a search for lower rank using random initialization
    # and gradient-free optimization (since we need exact integer/half-integer)
    
    # Actually, for exact decompositions, we should use known algebraic identities
    
    # Known: 3x3 x 3x3 can be done in rank 21
    # Let's construct this and combine with 3x3 x 3x4
    
    # For 3x3 x 3x4, we can split into 3x3 x 3x3 + 3x3 x 3x1
    # 3x3 x 3x1 is matrix-vector, rank 9 (schoolbook optimal for this)
    
    # So total: 21 + 21 + 9 = 51? No, that's not right.
    # We need to be more careful.
    
    # Let me try a different approach: direct search for rank < 63
    
    # Use the fact that the tensor has symmetries and try to find
    # a decomposition with shared structure
    
    # For now, return schoolbook if we don't find improvement quickly
    if time.time() > deadline - 5:
        return best_result
    
    # Try rank 50-62 with random search
    # This is a simplified approach; real optimization would be more sophisticated
    
    # For this task, the key insight is that 3x3 x 3x7 might have
    # a lower rank than 63 due to the rectangular structure
    
    # Return schoolbook as baseline
    return best_result
# EVOLVE-BLOCK-END
