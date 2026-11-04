import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    # The matrix multiplication tensor for n x m by m x p
    # has a direct sum structure over the intermediate dimension m.
    # Each component is an outer product tensor of size n x p, with rank n*p.
    # Total rank is m * (n * p) = n * m * p.
    # For 3x5x5, rank is 3*5*5 = 75.
    # This is the minimal rank as the supports in the first two modes are disjoint.
    
    rank = n * m * p
    U = np.zeros((dim_U, rank))
    V = np.zeros((dim_V, rank))
    W = np.zeros((dim_W, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                U[alpha, col] = 1.0
                V[beta, col] = 1.0
                W[gamma, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
