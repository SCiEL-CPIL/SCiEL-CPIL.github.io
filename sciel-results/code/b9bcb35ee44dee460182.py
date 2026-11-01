import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 seconds limit, keep margin
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Check time (should be very fast)
    if time.time() > deadline - 5:
        # Return best found (which is the current one)
        pass
        
    return {"U": U, "V": V, "W": W}
