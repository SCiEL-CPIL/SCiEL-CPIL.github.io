"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor for matrix multiplication M_{n,m,p} has entries M_{ijk} = 1.
    # A rank-1 decomposition is given by U, V, W being all-ones vectors.
    # U has shape (n*m, 1), V has shape (m*p, 1), W has shape (p*n, 1).
    
    rank = 1
    U = [[1] for _ in range(n * m)]
    V = [[1] for _ in range(m * p)]
    W = [[1] for _ in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
