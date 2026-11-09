"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Generate schoolbook decomposition immediately
    # This is guaranteed to be valid and has rank n*m*p
    best_rank = n * m * p
    U_fb = np.zeros((nm, best_rank))
    V_fb = np.zeros((mp, best_rank))
    W_fb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, col] = 1
                V_fb[j*p + k, col] = 1
                W_fb[k*n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = [[float(U_fb[i, j]) for j in range(len(U_fb[0]))] for i in range(len(U_fb))]
    V_list = [[float(V_fb[i, j]) for j in range(len(V_fb[0]))] for i in range(len(V_fb))]
    W_list = [[float(W_fb[i, j]) for j in range(len(W_fb[0]))] for i in range(len(W_fb))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
