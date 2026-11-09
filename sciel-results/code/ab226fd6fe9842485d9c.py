   def _construct_rank_nm(n, m, p):
       r = n * m
       U = np.eye(r)
       V = np.zeros((m * p, r))
       W = np.zeros((p * n, r))
       
       for i in range(n):
           for j in range(m):
               col = i * m + j
               # V[j*p+k, col] = 1 for all k
               for k in range(p):
                   V[j * p + k, col] = 1
               # W[k*n+i, col] = 1 for all k
               for k in range(p):
                   W[k * n + i, col] = 1
               
       return U.tolist(), V.tolist(), W.tolist()
   