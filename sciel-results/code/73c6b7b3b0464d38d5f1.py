   def _local_search_bitmask(adj, n, time_limit):
       # Precompute
       neighbors = [0]*n
       comp_neighbors = [0]*n
       for i in range(n):
           for j in range(n):
               if i != j and adj[i][j]:
                   neighbors[i] |= (1 << j)
               elif i != j:
                   comp_neighbors[i] |= (1 << j)
       
       def count_violations():
           k5 = 0
           ind5 = 0
           for s in itertools.combinations(range(n), 5):
               u,v,a,b,c = s
               # Check K5
               if (neighbors[u] & (1<<v) & (1<<a) & (1<<b) & (1<<c) and
                   neighbors[v] & (1<<u) & (1<<a) & (1<<b) & (1<<c) and
                   neighbors[a] & (1<<u) & (1<<v) & (1<<b) & (1<<c) and
                   neighbors[b] & (1<<u) & (1<<v) & (1<<a) & (1<<c) and
                   neighbors[c] & (1<<u) & (1<<v) & (1<<a) & (1<<b)):
                   k5 += 1
               # Check Ind5
               if (comp_neighbors[u] & (1<<v) & (1<<a) & (1<<b) & (1<<c) and
                   comp_neighbors[v] & (1<<u) & (1<<a) & (1<<b) & (1<<c) and
                   comp_neighbors[a] & (1<<u) & (1<<v) & (1<<b) & (1<<c) and
                   comp_neighbors[b] & (1<<u) & (1<<v) & (1<<a) & (1<<c) and
                   comp_neighbors[c] & (1<<u) & (1<<v) & (1<<a) & (1<<b)):
                   ind5 += 1
           return k5 + ind5
           
       violations = count_violations()
       if violations == 0: return True
       
       temp = 1.0
       start = time.time()
       
       while time.time() - start < time_limit:
           u = random.randint(0, n-1)
           v = random.randint(0, n-1)
           if u == v: continue
           
           # Identify affected 5-sets
           # We just recalculate total violations? No, that's O(n^5).
           # We need to only check affected sets.
           
           # List of 5-sets containing u and v
           affected_sets = []
           for a in range(n):
               if a == u or a == v: continue
               for b in range(n):
                   if b == u or b == v or b == a: continue
                   for c in range(n):
                       if c == u or c == v or c == a or c == b: continue
                       affected_sets.append((u,v,a,b,c))
           
           old_violations = 0
           for s in affected_sets:
               u,v,a,b,c = s
               if (neighbors[u] & (1<<v) & (1<<a) & (1<<b) & (1<<c) and
                   neighbors[v] & (1<<u) & (1<<a) & (1<<b) & (1<<c) and
                   neighbors[a] & (1<<u) & (1<<v) & (1<<b) & (1<<c) and
                   neighbors[b] & (1<<u) & (1<<v) & (1<<a) & (1<<c) and
                   neighbors[c] & (1<<u) & (1<<v) & (1<<a) & (1<<b)):
                   old_violations += 1
               if (comp_neighbors[u] & (1<<v) & (1<<a) & (1<<b) & (1<<c) and
                   comp_neighbors[v] & (1<<u) & (1<<a) & (1<<b) & (1<<c) and
                   comp_neighbors[a] & (1<<u) & (1<<v) & (1<<b) & (1<<c) and
                   comp_neighbors[b] & (1<<u) & (1<<v) & (1<<a) & (1<<c) and
                   comp_neighbors[c] & (1<<u) & (1<<v) & (1<<a) & (1<<b)):
                   old_violations += 1
                   
           # Flip
           adj[u][v] = not adj[u][v]
           adj[v][u] = not adj[v][u]
           
           # Update masks
           if adj[u][v]:
               neighbors[u] |= (1 << v)
               neighbors[v] |= (1 << u)
               comp_neighbors[u] &= ~(1 << v)
               comp_neighbors[v] &= ~(1 << u)
           else:
               neighbors[u] &= ~(1 << v)
               neighbors[v] &= ~(1 << u)
               comp_neighbors[u] |= (1 << v)
               comp_neighbors[v] |= (1 << u)
           
           new_violations = 0
           for s in affected_sets:
               u,v,a,b,c = s
               if (neighbors[u] & (1<<v) & (1<<a) & (1<<b) & (1<<c) and
                   neighbors[v] & (1<<u) & (1<<a) & (1<<b) & (1<<c) and
                   neighbors[a] & (1<<u) & (1<<v) & (1<<b) & (1<<c) and
                   neighbors[b] & (1<<u) & (1<<v) & (1<<a) & (1<<c) and
                   neighbors[c] & (1<<u) & (1<<v) & (1<<a) & (1<<b)):
                   new_violations += 1
               if (comp_neighbors[u] & (1<<v) & (1<<a) & (1<<b) & (1<<c) and
                   comp_neighbors[v] & (1<<u) & (1<<a) & (1<<b) & (1<<c) and
                   comp_neighbors[a] & (1<<u) & (1<<v) & (1<<b) & (1<<c) and
                   comp_neighbors[b] & (1<<u) & (1<<v) & (1<<a) & (1<<c) and
                   comp_neighbors[c] & (1<<u) & (1<<v) & (1<<a) & (1<<b)):
                   new_violations += 1
                   
           delta = new_violations - old_violations
           violations += delta
           
           if violations == 0:
               return True
               
           if delta > 0 and random.random() > pow(2.718, -delta/temp):
               # Revert
               adj[u][v] = not adj[u][v]
               adj[v][u] = not adj[v][u]
               if adj[u][v]:
                   neighbors[u] |= (1 << v)
                   neighbors[v] |= (1 << u)
                   comp_neighbors[u] &= ~(1 << v)
                   comp_neighbors[v] &= ~(1 << u)
               else:
                   neighbors[u] &= ~(1 << v)
                   neighbors[v] &= ~(1 << u)
                   comp_neighbors[u] |= (1 << v)
                   comp_neighbors[v] |= (1 << u)
               violations -= delta
               
           temp *= 0.999
           if temp < 0.01: temp = 0.01
           
       return False
   