import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li * m + j0 + lj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0 + lj) * p + k0 + lk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # 4 terms for k=4
    for j in range(m):
        for i in range(n):
            U[i * m + j][col] = 1
        V[j * p + 4][col] = 1
        for i in range(n):
            W[4 * n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
