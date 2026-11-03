"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Standard decomposition of the matrix multiplication tensor.
    # T[i, k, j] = 1 for all i, k, j.
    # Rank R = m.
    # U[i, k, r] = 1 if k == r else 0
    # V[k, j, r] = 1 if k == r else 0
    # W[j, i, r] = 1
    # Sum_r U[i,k,r] V[k,j,r] W[j,i,r] = Sum_r [k=r] [k=r] 1 = 1.
    
    R = m
    
    # U shape: (n * m, R)
    # Index: i * m + k
    U = [[0] * R for _ in range(n * m)]
    
    # V shape: (m * p, R)
    # Index: k * p + j
    V = [[0] * R for _ in range(m * p)]
    
    # W shape: (p * n, R)
    # Index: j * n + i
    W = [[0] * R for _ in range(p * n)]
    
    for r in range(R):
        # Fill U: U[i, k, r] = 1 if k == r
        for i in range(n):
            for k in range(m):
                if k == r:
                    idx = i * m + k
                    U[idx][r] = 1
                    
        # Fill V: V[k, j, r] = 1 if k == r
        for k in range(m):
            if k == r:
                for j in range(p):
                    idx = k * p + j
                    V[idx][r] = 1
                    
        # Fill W: W[j, i, r] = 1 for all i, j
        for j in range(p):
            for i in range(n):
                idx = j * n + i
                W[idx][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
