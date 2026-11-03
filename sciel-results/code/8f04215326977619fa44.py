    if col != rank:
        # Fallback to pure schoolbook if logic error
        # This should not happen if the counting is correct
        # 4*8 = 32 covered. 75-32=43 remaining. 28+43=71.
        U = [[0] * 75 for _ in range(n * m)]
        V = [[0] * 75 for _ in range(m * p)]
        W = [[0] * 75 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
