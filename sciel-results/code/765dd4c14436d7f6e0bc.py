"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook decomposition
    # U[i*m + j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p + k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n + i, c] = 1 if c corresponds to (i,j,k)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Check deadline (should never trigger for this small size)
    if time.time() > deadline:
        # Return best effort (which is the full schoolbook)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
