"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    # Build the 9x9x9 matrix multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # We know the rank is at most 23. Let's try to find a rank-23 decomposition.
    # Use a greedy approach: start with the schoolbook decomposition and try to reduce.
    
    # For now, use the schoolbook decomposition of rank 27
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = [[U[i,j] for j in range(rank)] for i in range(n*m)]
    V_list = [[V[i,j] for j in range(rank)] for i in range(m*p)]
    W_list = [[W[i,j] for j in range(rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
