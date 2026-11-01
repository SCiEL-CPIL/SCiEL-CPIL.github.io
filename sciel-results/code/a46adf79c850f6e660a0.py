   def construct():
       n, m, p = 2, 5, 6
       rank = 54
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       
       col = 0
       
       # Strassen helper
       def add_strassen_block(j_start, k_start):
           nonlocal col
           # 7 terms
           # T1
           U[0*m + j_start][col] += 1
           U[1*m + j_start+1][col] += 1
           V[j_start*p + k_start][col] += 1
           V[(j_start+1)*p + (k_start+1)][col] += 1
           W[k_start*n + 0][col] += 1
           W[(k_start+1)*n + 1][col] += 1
           col += 1
           # ... etc
           
       # Block 1: j in [0,1]
       # k in [0,1]
       add_strassen_block(0, 0)
       # k in [2,3]
       add_strassen_block(0, 2)
       # k in [4,5]
       add_strassen_block(0, 4)
       
       # Block 2: j in [2,3]
       add_strassen_block(2, 0)
       add_strassen_block(2, 2)
       add_strassen_block(2, 4)
       
       # Block 3: j in [4]
       for k in range(6):
           # i=0
           U[0*m + 4][col] = 1
           V[4*p + k][col] = 1
           W[k*n + 0][col] = 1
           col += 1
           # i=1
           U[1*m + 4][col] = 1
           V[4*p + k][col] = 1
           W[k*n + 1][col] = 1
           col += 1
           
       return {"U": U, "V": V, "W": W}
   