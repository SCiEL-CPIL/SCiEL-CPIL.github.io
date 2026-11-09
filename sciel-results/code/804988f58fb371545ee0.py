"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    # Schoolbook decomposition of rank n*m*p
    # For 2x3x7, rank is 42.
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices using numpy for efficiency
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, column] = 1
                V[v_idx, column] = 1
                W[w_idx, column] = 1
                
                column += 1
                
                if time.time() - start_time > 100:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
