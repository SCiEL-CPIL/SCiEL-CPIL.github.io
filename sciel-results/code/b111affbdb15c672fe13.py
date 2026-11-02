"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Strategy:
    Standard Schoolbook decomposition.
    Rank = 4 * 4 * 9 = 144.
    This is guaranteed to be correct and fast.
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Dimensions for factor matrices
    # U: (n*m) x rank = 16 x 144
    # V: (m*p) x rank = 36 x 144
    # W: (p*n) x rank = 36 x 144
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(m):
            for l in range(p):
                # U index: i*m + k
                u_idx = i * m + k
                # V index: k*p + l
                v_idx = k * p + l
                # W index: l*n + i (Trace convention W[k*n+i,r] where k is col of C, i is row of C)
                # C is n x p. C[i, l].
                # Convention: W[col*n + row, rank] ?
                # "Output factor follows the official trace convention W[k*n+i,r]"
                # Usually C = sum W_r U_r V_r.
                # C[i, j] = sum_r U[i*m+k_r] V[k_r*p+j] W[j*n+i]
                # So W index is j*n + i. Here j=l, i=i.
                w_idx = l * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
