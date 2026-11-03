import numpy as np
import time

def construct():
      n, m, p = 4, 4, 7
      rank = n * m * p
      # Schoolbook
      U = np.zeros((n*m, rank))
      V = np.zeros((m*p, rank))
      W = np.zeros((p*n, rank))
      col = 0
      for i in range(n):
          for j in range(m):
              for k in range(p):
                  U[i*m+j, col] = 1
                  V[j*p+k, col] = 1
                  W[k*n+i, col] = 1
                  col += 1
      return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
