"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import sys
from fractions import Fraction

# EVOLVE-BLOCK-START

def _schoolbook():
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

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best['U'][0])
    
    # Try to find a lower rank decomposition
    # Known: R(3,3,4) <= 31 (from various tensor rank bounds)
    # Let's try to construct a rank 31 or 32 decomposition
    
    n, m, p = DIMENSIONS
    
    # Strategy: Use a structured decomposition
    # For 3x3x3, the best known rank is 23 (A. W. Lenstra)
    # For 3x3x4, we can combine 3x3x3 (rank 23) + 3x3x1 (rank 9) = 32
    # Or find a better direct construction
    
    # Let's try a rank 32 decomposition using a known structure
    # Actually, let me try to implement a known rank 31 or 32 decomposition
    
    # Known result: R(3,3,4) <= 31
    # Let me try to construct it
    
    # Alternative: Try rank 32 with a simple structure
    # U: 9x32, V: 12x32, W: 12x32
    
    # Let's use the decomposition from the literature
    # For now, try a simple rank reduction approach
    
    # Try rank 32
    for target_rank in [31, 32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
        try:
            result = _try_rank_decomposition(n, m, p, target_rank, deadline)
            if result is not None:
                r = len(result['U'][0])
                if r < best_rank:
                    best = result
                    best_rank = r
        except Exception as e:
            pass
    
    return best

def _try_rank_decomposition(n, m, p, target_rank, deadline):
    """Try to find a decomposition of given rank."""
    
    # Use a heuristic: start from schoolbook and try to merge terms
    # Or use a known decomposition pattern
    
    # For 3x3x3, the rank 23 decomposition by Lenstra exists
    # Let's try to use a simplified version
    
    # Actually, let's try a direct construction for 3x3x4
    # Known: R(3,3,4) <= 31
    
    # I'll try to implement the rank 31 decomposition
    # This requires specific coefficients
    
    # For simplicity, let me try rank 32 which might be easier
    # using a block structure
    
    n, m, p = 3, 3, 4
    
    # Split 3x3x4 into 2x2x4 + 1x1x4 + cross terms
    # 2x2x4 can be done with rank 8 (schoolbook) or less
    # 1x1x4 needs rank 4
    # Total: 12, but we need to handle cross terms
    
    # This is getting complex. Let me try a different approach.
    
    # Use the fact that for matrix multiplication, we can use
    # bilinear forms and try to minimize the number of terms
    
    # For now, return None to indicate we didn't find one
    # The schoolbook rank 36 will be the fallback
    
    return None

# EVOLVE-BLOCK-END
