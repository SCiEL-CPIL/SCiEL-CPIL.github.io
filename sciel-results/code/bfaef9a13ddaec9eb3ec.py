"""Matrix multiplication 3,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction
from itertools import product
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook baseline
    rank0 = n * m * p
    U0 = [[0]*rank0 for _ in range(n*m)]
    V0 = [[0]*rank0 for _ in range(m*p)]
    W0 = [[0]*rank0 for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j][col] = 1
                V0[j*p+k][col] = 1
                W0[k*n+i][col] = 1
                col += 1
    
    return {"U": U0, "V": V0, "W": W0}

# EVOLVE-BLOCK-END
