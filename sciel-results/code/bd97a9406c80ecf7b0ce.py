"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank is 2 * p
    # Terms are indexed by (k, t) where k in [0, p) and t in [0, m)
    # Here m=2, so t in {0, 1}
    # Total rank = m * p
    rank = m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill the matrices based on the derived construction
    for k in range(p):
        for t in range(m):
            col = t * p + k
            
            # U vector: length n*m
            # U[i*m + j, col] = 1 if j == t, else 0
            for i in range(n):
                U[i * m + t][col] = 1
                
            # V vector: length m*p
            # V[j*p + k', col] = 1 if j == t and k' == k, else 0
            V[t * p + k][col] = 1
            
            # W vector: length p*n
            # W[k'*n + i', col] = 1 if k' == k, else 0
            # Note: i' can be anything in [0, n)
            for i_prime in range(n):
                W[k * n + i_prime][col] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
