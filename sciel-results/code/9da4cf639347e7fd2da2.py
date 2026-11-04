import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If we broke early, the result might be invalid/incomplete.
    # However, for n=3, m=4, p=7, the loop is very fast (84 iterations).
    # The time guard is just for compliance with the directive.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
