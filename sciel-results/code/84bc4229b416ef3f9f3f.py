   import numpy as np

   def construct():
       n, m, p = 3, 4, 4
       rank = n * m * p # 48
       
       U = np.zeros((n * m, rank))
       V = np.zeros((m * p, rank))
       W = np.zeros((p * n, rank))
       
       idx = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   # U index: i*m + j
                   # V index: j*p + k
                   # W index: k*n + i
                   U[i * m + j, idx] = 1
                   V[j * p + k, idx] = 1
                   W[k * n + i, idx] = 1
                   idx += 1
                   
       return {'U': U, 'V': V, 'W': W}
   