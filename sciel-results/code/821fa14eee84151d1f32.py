def construct():
       import random
       import time
       
       start_time = time.time()
       deadline = start_time + 115
       
       n = 20
       all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
       best_edges = []
       
       while time.time() < deadline:
           random.shuffle(all_edges)
           adj = [[-1]*n for _ in range(n)]
           neighbors = [[] for _ in range(n)]
           current_edges = []
           
           for u, v in all_edges:
               valid_color = -1
               colors = list(range(20))
               random.shuffle(colors)
               
               for c in colors:
                   ok = True
                   # Optimization: check conflicts
                   # We need to check all w in N(u) and x in N(v) with edge (w,x)
                   for w in neighbors[u]:
                       for x in neighbors[v]:
                           if w != x and adj[w][x] != -1:
                               c1 = adj[u][w]
                               c2 = adj[w][x]
                               c3 = adj[x][v]
                               # Check distinctness
                               # c != c1, c != c2, c != c3
                               # c1 != c2, c1 != c3, c2 != c3
                               if c == c1 or c == c2 or c == c3 or c1 == c2 or c1 == c3 or c2 == c3:
                                   ok = False
                                   break
                       if not ok:
                           break
                   
                   if ok:
                       valid_color = c
                       break
               
               if valid_color != -1:
                   adj[u][v] = valid_color
                   adj[v][u] = valid_color
                   neighbors[u].append(v)
                   neighbors[v].append(u)
                   current_edges.append([u, v, valid_color])
           
           if len(current_edges) > len(best_edges):
               best_edges = current_edges
               
       return best_edges
