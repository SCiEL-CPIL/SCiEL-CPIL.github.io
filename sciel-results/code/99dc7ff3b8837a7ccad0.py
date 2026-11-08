# 4x4x1 part
for i in range(4):
    for j in range(4):
        U[i*5 + j][col] = 1
        V[j*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
