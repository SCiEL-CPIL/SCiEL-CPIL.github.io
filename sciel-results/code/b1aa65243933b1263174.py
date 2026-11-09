    def try_rank24():
        rank = m * p  # 24
        ...
        col = 0
        for j in range(m):
            for k in range(p):
                V[j * p + k][col] = 1
                for i in range(n):
                    U[i * m + j][col] = 1
                    W[k * n + i][col] = 1
                col += 1
