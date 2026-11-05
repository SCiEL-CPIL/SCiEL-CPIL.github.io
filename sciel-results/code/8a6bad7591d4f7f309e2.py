   for k1 in range(N):
       for k2 in range(k1+1, N):
           for v1 in range(N):
               for v2 in range(N):
                   if time.time() > deadline:
                       break
   