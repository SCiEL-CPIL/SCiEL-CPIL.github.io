"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (or best known) exact tensor decomposition for 5x5x5 matrix multiplication.
    Currently implements the schoolbook decomposition with rank 125.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check deadline
                if time.time() > deadline:
                    # Return partial or best known result
                    # In this case, we just break and return what we have, 
                    # but since it's a full loop, we should just finish fast.
                    # The loop is 125 iterations, so it's very fast.
                    pass
                
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
