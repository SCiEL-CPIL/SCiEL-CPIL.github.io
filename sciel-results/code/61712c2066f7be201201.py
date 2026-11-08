           for i in range(n):
               for a in range(m):
                   for k in range(p):
                       acc = 0
                       for c in range(rank):
                           acc += U[i*m+a][c] * V[a*p+k][c] * W[k*n+i][c]
                       if acc != 1:
                           return False
           