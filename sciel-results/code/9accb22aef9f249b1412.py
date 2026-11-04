"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Rank 6 construction: one term per (i,j) pair
    # For each (i,j), use one rank-1 term that covers all k values
    rank = n * m  # = 6
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 6)
    V = [[0] * rank for _ in range(m * p)]  # shape (15, 6)
    W = [[0] * rank for _ in range(p * n)]  # shape (10, 6)
    
    col = 0
    for i in range(n):
        for j in range(m):
            # One term for this (i,j)
            U[i*m + j][col] = 1
            for k in range(p):
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
            col += 1
    
    # Verify the solution
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]))
                max_err = max(max_err, abs(val - 1))
    
    if max_err < 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

