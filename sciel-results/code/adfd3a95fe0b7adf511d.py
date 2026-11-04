"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 9 matrix.
    
    The rank is 54 (schoolbook algorithm).
    U shape: (n*m, rank) = (6, 54)
    V shape: (m*p, rank) = (27, 54)
    W shape: (p*n, rank) = (18, 54)
    
    The decomposition satisfies:
    sum_r U_{i*m+j, r} * V_{j'*p+k', r} * W_{k''*n+i'', r} = delta_{i,i''} * delta_{j,j'} * delta_{k,k''}
    """
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][column] = 1
                # V index: j*p + k
                V[j * p + k][column] = 1
                # W index: k*n + i
                W[k * n + i][column] = 1
                column += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
