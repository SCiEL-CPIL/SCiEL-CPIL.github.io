   for i in range(n):
       for j in range(m):
           for k in range(p):
               s = 0.0
               for r_idx in range(r):
                   s += U[i * m + j][r_idx] * V[j * p + k][r_idx] * W[k * n + i][r_idx]
               if abs(s - 1.0) > tol:
                   return False
   