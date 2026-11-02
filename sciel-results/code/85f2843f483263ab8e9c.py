    for j in range(3):
        for i in range(3):
            U[i*3+j][col] = 1
            W[6*3+i][col] = 1
        V[j*3+6][col] = 1
        col += 1
