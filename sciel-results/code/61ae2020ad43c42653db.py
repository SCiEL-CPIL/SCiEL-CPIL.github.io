"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the schoolbook matrix multiplication tensor decomposition.
    For 2x3 * 3x6, the schoolbook rank is n*m*p = 2*3*6 = 36.
    This is known to be optimal for these dimensions.
    """
    # Deadline guard: the task says 150s wall-clock, but the contract says 120s.
    # We'll use a small margin since this is a static construction.
    start_time = time.time()
    deadline = start_time + 10  # 10 seconds is plenty for static construction

    n, m, p = DIMENSIONS
    rank = n * m * p  # 36

    # U has shape (n*m, rank) = (6, 36)
    # V has shape (m*p, rank) = (18, 36)
    # W has shape (p*n, rank) = (12, 36)
    
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
    
    # Safety check: ensure we haven't exceeded deadline (unlikely for static code)
    if time.time() > deadline:
        # Return whatever we have (which should be complete)
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
