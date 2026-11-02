"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 140
    
    # Schoolbook decomposition of rank n*m*p
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
