    def construct():
        n, m, p = 2, 3, 3
        rank = 17
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # ...
        # Part 1
        U[0][col]=1; U[3][col]=1
        V[0][col]=1; V[5][col]=1
        W[0][col]=1; W[2][col]=1; W[1][col]=1; W[3][col]=1
        col+=1
        # ...
    