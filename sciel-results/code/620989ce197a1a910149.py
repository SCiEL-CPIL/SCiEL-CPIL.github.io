    col_offset = rank_444
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i * 4 + j][col_offset + col] = 1
                V[j * 7 + k][col_offset + col] = 1
                W[k * 4 + i][col_offset + col] = 1
                col += 1
