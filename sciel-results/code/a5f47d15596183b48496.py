"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition of rank n*m*p.
    Includes a deadline guard to ensure it finishes within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
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
                if time.time() > deadline:
                    # If we exceed time, break out of all loops.
                    # Since this loop is fast, this shouldn't happen, but it's a safety guard.
                    # Returning partial results would be invalid, so we rely on the loop finishing.
                    # If it doesn't, the result will be invalid, but we avoid a hard timeout crash.
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            else:
                continue
            break
        else:
            continue
        break
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
