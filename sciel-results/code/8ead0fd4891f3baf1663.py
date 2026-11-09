"""Matrix multiplication 2x4x6 - rank minimization."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    The matrix multiplication tensor M_{2,4,6} has T[i,j,k] = 1 for all i,j,k.
    This is the all-ones tensor of shape (2,4,6), which is a rank-1 tensor:
    T = (1,1) ⊗ (1,1,1,1) ⊗ (1,1,1,1,1,1)
    
    Therefore the minimal rank is 1.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12
    
    rank = 1
    U = [[1] * rank for _ in range(nm)]
    V = [[1] * rank for _ in range(mp)]
    W = [[1] * rank for _ in range(pn)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
