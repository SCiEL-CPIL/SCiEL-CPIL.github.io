import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
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
                
                # Check deadline
                if time.time() > deadline:
                    # Return what we have? No, schoolbook is fast.
                    # If we were searching, we would return best found.
                    # Here, schoolbook is deterministic and fast.
                    pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
