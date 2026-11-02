"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Indices: U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Target T[i, j, k] = 1 for all i, j, k

    # 1. Generate Schoolbook Decomposition (Rank 36)
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb), dtype=int)
    V = np.zeros((m * p, rank_sb), dtype=int)
    W = np.zeros((p * n, rank_sb), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
