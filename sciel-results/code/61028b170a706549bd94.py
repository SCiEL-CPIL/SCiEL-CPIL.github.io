import time

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0

def construct():
    """
    Constructs a tensor decomposition for 3x3 by 3x8 matrix multiplication.
    Uses the standard schoolbook algorithm with rank 72.
    Includes a deadline guard to prevent timeout.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                elapsed = time.time() - start_time
                if elapsed > DEADLINE:
                    break
            if elapsed > DEADLINE:
                break
        if elapsed > DEADLINE:
            break
    
    return {"U": U, "V": V, "W": W}
