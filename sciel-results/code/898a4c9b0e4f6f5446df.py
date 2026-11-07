   import sys
   import time
   import random
   from collections import defaultdict

   P = { ... } # Keep original P

   def construct():
       n = P['n']
       edges = P['edges']
       
       # Build global adjacency
       adj = defaultdict(list)
       for u, v in edges:
           adj[u].append(v)
           adj[v].append(u)
           
       # Function to find M edge-disjoint Hamiltonian cycles on subset S
       def find_m_cycles(S, M):
           S_set = set(S)
           # Filter edges
           S_edges = []
           for u, v in edges:
               if u in S_set and v in S_set:
                   S_edges.append((u, v))
                   
           # If not enough edges, return None
           if len(S_edges) < M * len(S):
               return None
               
           # Local adjacency
           l_adj = defaultdict(list)
           for u, v in S_edges:
               l_adj[u].append(v)
               l_adj[v].append(u)
               
           used_edges = set()
           cycles = []
           
           # DFS to find one cycle
           def find_one_cycle():
               # Start at min(S)
               start = min(S)
               path = [start]
               visited = {start}
               
               def dfs(current):
                   if len(path) == len(S):
                       if is_avail(current, start):
                           return list(path)
                       return None
                   
                   # Heuristic: sort neighbors by degree in remaining graph?
                   # For now, randomize or standard
                   neighbors = l_adj[current]
                   random.shuffle(neighbors) # To avoid getting stuck in local minima
                   
                   for nb in neighbors:
                       if nb not in visited:
                           if is_avail(current, nb):
                               visited.add(nb)
                               path.append(nb)
                               res = dfs(nb)
                               if res:
                                   return res
                               path.pop()
                               visited.remove(nb)
                   return None
                   
               def is_avail(u, v):
                   if u > v: u, v = v, u
                   return (u, v) not in used_edges
               
               return dfs(start)
               
           for _ in range(M):
               c = find_one_cycle()
               if not c:
                   return None
               cycles.append(c)
               for i in range(len(c)):
                   u, v = c[i], c[(i+1)%len(c)]
                   if u > v: u, v = v, u
                   used_edges.add((u, v))
                   
           return cycles

       # Try to find 5 cycles
       # Sizes to try: 14, 13, 12, 11
       # Try random subsets
       best_cycles = None
       
       # First, try the full set for 3 cycles (baseline)
       # We know it works, so we have a fallback.
       # But let's try to beat it.
       
       # Try 5 cycles on subsets of size 11-14
       for size in [11, 12, 13, 14]:
           found = False
           # Try 100 random subsets
           for _ in range(100):
               S = random.sample(range(n), size)
               res = find_m_cycles(S, 5)
               if res:
                   best_cycles = res
                   found = True
                   break
           if found:
               return best_cycles
               
       # Try 4 cycles on subsets of size 9-10
       for size in [9, 10]:
           found = False
           for _ in range(100):
               S = random.sample(range(n), size)
               res = find_m_cycles(S, 4)
               if res:
                   best_cycles = res
                   found = True
                   break
           if found:
               return best_cycles
               
       # Fallback: Full set, 3 cycles
       # Use the original logic or find_m_cycles(V, 3)
       res = find_m_cycles(list(range(n)), 3)
       if res:
           return res
           
       # If even 3 fails (unlikely), return empty? No, must be non-empty.
       # But 3 is guaranteed in the seed.
       
       return []
   