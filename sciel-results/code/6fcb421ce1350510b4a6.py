import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 110  # 10 seconds before the 120s limit
    
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
    
    # Check if we're close to the deadline
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a better decomposition
    # ... (search code here)
    
    return {"U": U, "V": V, "W": W}
