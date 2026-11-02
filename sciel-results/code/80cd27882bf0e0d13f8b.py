DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    rank = n * m * p  # 144
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 144)
    V = [[0] * rank for _ in range(m * p)]  # (36, 144)
    W = [[0] * rank for _ in range(p * n)]  # (36, 144)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
