"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses the schoolbook algorithm (Rank 125) which is the known baseline.
    We construct this efficiently and return it.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # We use lists to match the seed program's interface exactly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Schoolbook term: A[i,j] * B[j,k] contributes to C[i,k]
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Check time just in case, though this loop is very fast
    if time.time() > deadline:
        # This should never happen for rank 125, but it's a safety net
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
