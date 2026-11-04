   import numpy as np

   DIMENSIONS = [2, 5, 5]

   def construct():
       n, m, p = DIMENSIONS
       rank = n * m * p
       
       # Generate indices for the trivial decomposition
       # The trivial decomposition corresponds to the basis of the tensor space
       
       # We can construct U, V, W by iterating over the basis elements
       # This is structurally different from the list-of-lists approach
       
       U = np.zeros((n * m, rank), dtype=int)
       V = np.zeros((m * p, rank), dtype=int)
       W = np.zeros((p * n, rank), dtype=int)
       
       # Map (i, j, k) to column index
       # col = i * m * p + j * p + k
       
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   col = i * m * p + j * p + k
                   U[i * m + j, col] = 1
                   V[j * p + k, col] = 1
                   W[k * n + i, col] = 1
                   
       return {"U": U, "V": V, "W": W}
   