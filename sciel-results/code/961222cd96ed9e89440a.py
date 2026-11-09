rank = n * m * p  # 60
U = [[0] * rank for _ in range(n * m)]  # 6×60
V = [[0] * rank for _ in range(m * p)]  # 30×60
W = [[0] * rank for _ in range(p * n)]  # 20×60

col = 0
for i in range(n):
    for j in range(m):
        for k in range(p):
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
