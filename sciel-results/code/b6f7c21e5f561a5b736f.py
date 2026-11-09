"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction
from itertools import product

import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # The multiplication is A(2x3) * B(3x6) = C(2x6)
    # We can split B into 3 blocks of 3x2: B0, B1, B2
    # C = [C0, C1, C2] where Ci = A * Bi
    # Each Ci is 2x2.
    # The rank of 2x3x2 multiplication is 3.
    # So total rank is 3 * 3 = 9.
    
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # For each block b in 0..2 (corresponding to k in 2b, 2b+1)
    for b in range(3):
        # The 3 terms for this block correspond to j=0, 1, 2
        for j in range(3):
            col = b * 3 + j
            
            # U: 1 at rows i*m + j for i=0,1
            for i in range(n):
                U[i * m + j][col] = 1
                
            # V: 1 at rows j*p + k for k=2b, 2b+1
            for k in range(2 * b, 2 * b + 2):
                V[j * p + k][col] = 1
                
            # W: 1 at rows k*n + i for all i,k in this block
            for i in range(n):
                for k in range(2 * b, 2 * b + 2):
                    W[k * n + i][col] = 1
                    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
