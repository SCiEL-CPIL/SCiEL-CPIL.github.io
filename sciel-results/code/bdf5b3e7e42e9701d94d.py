"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n, m, p = DIMENSIONS
    
    # Target rank based on the insight that n=2 allows pairing
    # of the two i-dimensions for each (j,k) pair.
    # Total (j,k) pairs = m*p = 25.
    rank = m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct the decomposition
    # For each j in 0..m-1 and k in 0..p-1, assign a unique rank r
    for j in range(m):
        for k in range(p):
            r = j * p + k
            
            # V[j*p+k, r] = 1
            V[j * p + k][r] = 1
            
            # For i=0: U[0*m+j, r] = 1, W[k*n+0, r] = 1
            U[0 * m + j][r] = 1
            W[k * n + 0][r] = 1
            
            # For i=1: U[1*m+j, r] = 1, W[k*n+1, r] = 1
            U[1 * m + j][r] = 1
            W[k * n + 1][r] = 1
            
            # Check time just in case, though this is O(1) per step
            if time.time() > deadline:
                # This should never happen for rank 25, but good practice
                pass
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
