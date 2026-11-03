"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    deadline = time.time() + 130  # 130s limit, 10s margin

    # We will use the schoolbook decomposition as the baseline
    rank_schoolbook = n * m * p  # 100
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][column] = 1
                V_schoolbook[j * p + k][column] = 1
                W_schoolbook[k * n + i][column] = 1
                column += 1

    # Try to find a better decomposition using bilinear optimization
    # We'll use a randomized approach with local search
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook

    # Initialize with small integers and try to optimize
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We want U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over c to equal T[i,j,k]
    
    # Let's try a rank-100 solution and then try to reduce
    # Actually, let's start with a known good solution and try to improve
    
    # For now, return the schoolbook solution
    return {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
# EVOLVE-BLOCK-END
