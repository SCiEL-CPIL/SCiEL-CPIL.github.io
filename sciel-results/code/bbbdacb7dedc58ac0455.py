"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use standard schoolbook decomposition for guaranteed correctness
    # Rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U: i*m + j
                u_idx = i * m + j
                # Index in V: j*p + k
                v_idx = j * p + k
                # Index in W: k*n + i
                w_idx = k * n + i
                
                U[u_idx][r_idx] = 1
                V[v_idx][r_idx] = 1
                W[w_idx][r_idx] = 1
                
                r_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
