import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 3, 10
    rank = n * m * p  # 60
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 60
    V = [[0] * rank for _ in range(m * p)]  # 30 x 60
    W = [[0] * rank for _ in range(p * n)]  # 20 x 60
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    # Return what we have so far (shouldn't happen with rank 60)
                    return {"U": U, "V": "V", "W": W}
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][column] = 1
                V[b][column] = 1
                W[c][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
