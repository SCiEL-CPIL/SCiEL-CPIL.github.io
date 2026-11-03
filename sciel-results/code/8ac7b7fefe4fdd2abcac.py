"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Added deadline guard to ensure completion within time limits.
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
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
                
                # Check deadline periodically
                if column % 10 == 0 and time.time() > deadline:
                    # If we run out of time, return what we have (which might be incomplete)
                    # But for schoolbook, it's very fast, so this shouldn't trigger.
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
    
    # If we broke early due to time, the decomposition might be invalid.
    # However, 84 iterations is negligible compared to 120s.
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
