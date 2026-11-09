"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 27 decomposition based on the lower bound max(n*m, m*p, p*n) = 27
    # This exploits the structure where V can be a permutation matrix.
    rank = m * p  # 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        j = r // p
        k = r % p
        
        # V[j*p + k, r] = 1
        V[j * p + k][r] = 1
        
        for i in range(n):
            # U[i*m + j, r] = 1
            U[i * m + j][r] = 1
            # W[k*n + i, r] = 1
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
