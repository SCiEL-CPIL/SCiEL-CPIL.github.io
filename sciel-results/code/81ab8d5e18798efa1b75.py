   for i in range(n):
       for k in range(m):
           for j in range(p):
               a = i*m + k
               b = k*p + j
               c = j*n + i
               target[a, b, c] = 1.0
   