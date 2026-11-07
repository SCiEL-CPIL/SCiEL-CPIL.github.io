   for u in adj[v]:
       if color[u] != -1:
           cu = color[u]
           if cu == c:
               ok = False
               break
           key = (c, cu) if c < cu else (cu, c)
           if key in uf:
               # Safe find
               def safe_find(x):
                   if x not in uf[key]:
                       return x
                   root = x
                   while uf[key][root] != root:
                       root = uf[key][root]
                   return root
               if safe_find(v) == safe_find(u):
                   ok = False
                   break
   