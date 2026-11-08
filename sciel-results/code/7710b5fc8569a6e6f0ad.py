"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p  # 48
    
    # We'll try to find a lower rank decomposition.
    # The known lower bound for matrix multiplication M_{n,m,p} is max(n*m, m*p, n*p) = max(6, 24, 16) = 24.
    # But we need an exact integer/half-integer decomposition.
    
    # For now, let's stick with the schoolbook decomposition which is guaranteed to work.
    # The task says "rank-minimal" but the score is just the rank value.
    # So we want the smallest possible rank.
    
    # Let's try a few different approaches to potentially get a lower rank.
    
    # Approach 1: Schoolbook (always works)
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
