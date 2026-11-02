   import random
   import numpy as np
   from collections import deque

   def construct(k, g):
       if k == 4 and g == 10:
           return _construct_4_10_cage()
       else:
           return _construct_general(k, g)

   def _construct_4_10_cage():
       n = 320
       # Try to find a 4-regular graph of girth 10 on 320 vertices
       # Use a fixed seed for reproducibility
       random.seed(42)
       
       for attempt in range(1000): # Limit attempts
           # Generate random 4-regular graph
           adj = _generate_random_regular_graph(n, 4)
           if adj is None:
               continue
           
           # Check girth
           girth = _compute_girth(adj, n)
           if girth >= 10:
               edges = []
               for i in range(n):
                   for j in range(i+1, n):
                       if adj[i][j]:
                           edges.append((i, j))
               return n, edges
       
       # Fallback: Return a larger valid graph if 320 not found
       # Or raise error? No, must return valid.
       # Construct a valid graph with n > 320
       return _construct_general(4, 10)

   def _generate_random_regular_graph(n, k):
       # Configuration model
       stubs = []
       for i in range(n):
           for _ in range(k):
               stubs.append(i)
       random.shuffle(stubs)
       
       adj = [[0]*n for _ in range(n)]
       for i in range(0, n*k, 2):
           u = stubs[i]
           v = stubs[i+1]
           if u == v:
               return None # Self loop
           if adj[u][v]:
               return None # Multi edge
           adj[u][v] = 1
           adj[v][u] = 1
       return adj

   def _compute_girth(adj, n):
       # BFS from each vertex
       min_girth = float('inf')
       for start in range(n):
           dist = [-1]*n
           parent = [-1]*n
           dist[start] = 0
           queue = deque([start])
           while queue:
               u = queue.popleft()
               for v in range(n):
                   if adj[u][v]:
                       if dist[v] == -1:
                           dist[v] = dist[u] + 1
                           parent[v] = u
                           queue.append(v)
                       elif v != parent[u]:
                           cycle_len = dist[u] + dist[v] + 1
                           if cycle_len < min_girth:
                               min_girth = cycle_len
       return min_girth
   