    school_U = [[0] * 12 for _ in range(4)]
    school_V = [[0] * 12 for _ in range(10)]
    school_W = [[0] * 12 for _ in range(10)]
    col = 0
    for i in range(2):
        for j in range(2):
            for k2 in range(3):
                school_U[i * 2 + j][col] = 1
                school_V[j * 5 + (k2 + 2)][col] = 1
                school_W[(k2 + 2) * 2 + i][col] = 1
                col += 1
