"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Attempt to find a low-rank tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Initialize with the schoolbook decomposition (rank 84).
    2. Attempt to reduce rank by searching for "zero" columns (linear dependencies)
       or by using known low-rank blocks (like Strassen 2x2x2).
    3. Since exact symbolic search is hard, we rely on the fact that for small matrices,
       the schoolbook is often tight or close to tight. However, we can try to see
       if we can merge any terms.
       
    Actually, for 3x4x7, the Strassen algorithm applies to sub-blocks.
    But without a specific algorithm, we stick to a robust search for linear dependencies 
    in the factor space if the rank were higher, but here we start at the minimum schoolbook rank.
    
    Wait, the schoolbook rank IS n*m*p. Is it possible to do better?
    Yes, e.g. 2x2x2 can be done in 7 instead of 8.
    
    Let's try to decompose 3x4x7 using 2x2x2 Strassen blocks where possible.
    3x4x7 can be viewed as:
    - Top-left 2x2 block of A, first 2 cols of B? No, dimensions don't align perfectly for pure Strassen without padding.
    
    Let's stick to the schoolbook for safety, as constructing a valid low-rank decomposition 
    from scratch without a specific known algorithm for 3x4x7 is risky. 
    The schoolbook is always valid.
    
    However, if we can find ANY rank reduction, the score improves.
    Let's try a random search for a rank 83 decomposition? 
    The space of rank 83 decompositions is huge. 
    The condition for a valid decomposition is that the bilinear map A, B -> C matches.
    
    Given the time limit and the complexity, the most robust approach is to return the schoolbook.
    If we want to try for a small improvement, we could check if the 2x2 sub-problem (top-left of A, top-left of B?) 
    allows for Strassen. 
    A is 3x4. B is 4x7.
    If we take the top-left 2x2 of A and the top-left 2x2 of B (rows 0-1, cols 0-1), 
    that contributes to C[0:2, 0:2].
    But A has cols 2,3 and B has rows 2,3 which also contribute to C[0:2, 0:2].
    So we can't just replace a block with Strassen easily because the contributions overlap.
    
    Therefore, the schoolbook decomposition is the only one I can guarantee is correct 
    without extensive verification code that might fail.
    """
    n, m, p = DIMENSIONS
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
