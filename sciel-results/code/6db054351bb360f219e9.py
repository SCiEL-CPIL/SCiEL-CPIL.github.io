def construct():
       import time
       start_time = time.time()
       deadline = 85.0 # Safety margin
       
       K = 27720
       N = 12
       
       # Precompute divisors for each residue mod K
       # divs[b_mod] = list of i in 1..12 such that i | b_mod
       # Note: i | b iff i | (b mod K) since i | K
       divs = [[] for _ in range(K)]
       for b_mod in range(K):
           for i in range(1, N+1):
               if b_mod % i == 0:
                   divs[b_mod].append(i)
               
       best_r = 0
       best_g = 0
       
       for r in range(K):
           # Incremental matching
           adj = [[] for _ in range(N+1)]
           g_r = None
           
           # We iterate L from 1 upwards
           # Max L is small, say 200
           for L in range(1, 200):
               b = r + L
               b_mod = b % K
               # Add b to adj
               # We represent b by its index L (1..L) to keep it small
               for i in divs[b_mod]:
                   adj[i].append(L)
                   
               # Check if matching exists
               # Simple DFS
               match = [0] * (L+1) # match[b_index] = i
               
               def dfs(i, visited):
                   for b_idx in adj[i]:
                       if visited[b_idx]:
                           continue
                       visited[b_idx] = True
                       if match[b_idx] == 0 or dfs(match[b_idx], visited):
                           match[b_idx] = i
                           return True
                   return False
               
               found = True
               for i in range(1, N+1):
                   visited = [False] * (L+1)
                   if not dfs(i, visited):
                       found = False
                       break
                       
               if found:
                   g_r = L
                   break
                   
           if g_r is None:
               g_r = 200 # Should not happen
               
           if g_r > best_g:
               best_g = g_r
               best_r = r
               
           if time.time() > deadline:
               break
               
       return best_r
