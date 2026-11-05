   def can_add(u, v, adj):
       if v in adj[u]:
           return False
       
       # 1. Check common neighbors of u and v
       if len(adj[u] & adj[v]) > 2:
           return False
       
       # 2. Check all other vertices w
       for w in range(n):
           if w == u or w == v:
               continue
           
           # Pair {u, w}
           if v in adj[w]:
               # v becomes a new common neighbor of u and w
               if len(adj[u] & adj[w]) > 1:
                   return False
           
           # Pair {v, w}
           if u in adj[w]:
               # u becomes a new common neighbor of v and w
               if len(adj[v] & adj[w]) > 1:
                   return False
       
       return True
   