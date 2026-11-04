def construct():
       n, m, p = 2, 3, 5
       rank = n * m * p
       
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       
       c = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   # U index: i*m + j
                   # V index: j*p + k
                   # W index: k*n + i
                   U[i * m + j][c] = 1
                   V[j * p + k][c] = 1
                   W[k * n + i][c] = 1
                   c += 1
                   
       return {"U": U, "V": V, "W": W}
