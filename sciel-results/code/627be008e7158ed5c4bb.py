# M7 = (a01-a11)*(b01+b11)
U[a01[0]][start_col] = 1
U[a11[0]][start_col] = -1
V[b01[0]][start_col] = 1
V[b11[0]][start_col] = 1
W[c00[0]][start_col] = 1
start_col += 1
