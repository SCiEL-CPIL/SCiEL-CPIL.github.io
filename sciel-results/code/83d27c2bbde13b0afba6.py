   for j in range(5):
       U[2][col] = 1; V[10 + j][col] = 1; W[j][col] = 1
       col += 1
       U[5][col] = 1; V[10 + j][col] = 1; W[5 + j][col] = 1
       col += 1
   