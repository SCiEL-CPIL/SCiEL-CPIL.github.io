    column = 0
    for block in range(2):
        k_offset = block * 3  # 0 for B1, 3 for B2
        for i in range(n):
            for j in range(m):
                for k in range(3):
                    U[i * m + j][column] = 1
                    V[j * p + (k_offset + k)][column] = 1
                    W[(k_offset + k) * n + i][column] = 1
                    column += 1
