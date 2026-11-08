"""Optimized program for matrix multiplication tensor decomposition 2x3 by 3x8."""
import time
import sys

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We need to find rank-minimal decomposition.
    # The schoolbook rank is n*m*p = 48.
    # Known lower bounds and constructions for small matrices:
    # For 2x3 by 3x8, we can potentially do better than 48.
    # 
    # Let's try to construct a decomposition with rank < 48.
    # One approach: exploit the structure.
    #
    # Actually, let's first try a simple improvement over schoolbook.
    # Schoolbook uses 48 terms.
    #
    # Let's try to find a decomposition of rank 42 or lower.
    # This is a research problem, but we can try some heuristics.
    #
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    # Rank = 48.
    
    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 6 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 16 x 48
    
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
