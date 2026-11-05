"""
Program for 5x5 matrix multiplication tensor decomposition.
Uses the schoolbook algorithm (rank 125) which is guaranteed to be exact.
Includes a deadline guard to ensure completion within the time limit.
"""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
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
                
                # Check time occasionally (though this loop is very fast)
                if column % 100 == 0 and time.time() > deadline:
                    # In this specific case, the loop is too fast to be an issue,
                    # but this structure allows for more complex searches if added.
                    break
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
