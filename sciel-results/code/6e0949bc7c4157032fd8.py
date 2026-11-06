   Au_row = A[u]
   Av_row = A[v]
   A2u = A2[u]
   A2v = A2[v]
   for w in range(N):
       A2u[w] += Av_row[w]
       A2v[w] += Au_row[w]
   