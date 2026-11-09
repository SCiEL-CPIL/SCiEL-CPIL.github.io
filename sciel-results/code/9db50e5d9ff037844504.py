        def construct():
            n, m, p = 4, 4, 9
            rank = 48
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            col = 0
            # Part 1: 4x4x4 for B cols 0-3
            for i in range(n):
                for j in range(m):
                    for k in range(4):
                        U[i*m + j][col] = 1
                        V[j*4 + k][col] = 1
                        W[k*n + i][col] = 1
                        col += 1
            # Part 2: 4x4x4 for B cols 4-7
            for i in range(n):
                for j in range(m):
                    for k in range(4):
                        U[i*m + j][col] = 1
                        V[j*4 + (k+4)][col] = 1
                        W[(k+4)*n + i][col] = 1
                        col += 1
            # Part 3: 4x4x1 for B col 8
            for i in range(n):
                for j in range(m):
                    U[i*m + j][col] = 1
                    V[j*4 + 8][col] = 1
                    W[8*n + i][col] = 1
                    col += 1
            return {"U": U, "V": V, "W": W}
        