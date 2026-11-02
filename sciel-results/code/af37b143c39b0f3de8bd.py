"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit, leaving 20s buffer for 120s total
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
                # Check time periodically
                if time.time() > deadline:
                    # If we hit the deadline, return what we have (which should be complete by now)
                    # Since rank is small (54), this loop is instantaneous.
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
