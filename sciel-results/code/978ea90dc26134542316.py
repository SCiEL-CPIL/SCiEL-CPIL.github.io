import time

DIMENSIONS = [2, 3, 6]

def construct():
    start = time.time()
    deadline = start + 120  # 120s limit, use 120s as deadline
    
    n, m, p = DIMENSIONS
    rank = 6
    U = [[1]*rank for _ in range(n*m)]
    V = [[1]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            W[k*n + i][k] = 1
    
    return {"U": U, "V": V, "W": W}
