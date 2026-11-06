   import time
   import random
   from itertools import combinations

   def construct():
       start_time = time.time()
       deadline = start_time + 145

       def random_graph(n, p, seed):
           rng = random.Random(seed)
           edges = []
           for u in range(n):
               for v in range(u + 1, n):
                   if rng.random() < p:
                       edges.append((u, v))
           return edges

       def solve(edges):
           if not edges:
               return []
           
           n = max(max(e) for e in edges) + 1
           adj = [set() for _ in range(n)]
           for u, v in edges:
               adj[u].add(v)
               adj[v].add(u)
           
           # 1. Greedy Partition
           remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
           cliques = []
           
           while remaining_edges:
               if time.time() > deadline:
                   # Fallback: add remaining edges as 2-cliques
                   for e in remaining_edges:
                       cliques.append(list(e))
                   break
               
               # Find best clique
               best_clique = []
               # ... (existing greedy logic) ...
               
               # Remove covered edges
               # ...
               
               cliques.append(best_clique)
               
           # 2. Merge Pass
           # Repeat until no merges
           merged = True
           while merged:
               merged = False
               new_cliques = []
               # Try to merge pairs
               i = 0
               while i < len(cliques):
                   C_i = cliques[i]
                   j = i + 1
                   merged_pair = False
                   while j < len(cliques):
                       C_j = cliques[j]
                       # Check if union is clique
                       # All vertices in C_i must be adjacent to all in C_j
                       is_clique = True
                       for u in C_i:
                           for v in C_j:
                               if v not in adj[u]:
                                   is_clique = False
                                   break
                           if not is_clique:
                               break
                       
                       if is_clique:
                           # Merge
                           C_i = C_i + C_j
                           # Remove C_j from list
                           cliques.pop(j)
                           merged = True
                           # Don't increment j, check new C_j with same C_i
                       else:
                           j += 1
                   
                   new_cliques.append(C_i)
                   i += 1
               cliques = new_cliques
               
           return cliques

       c1 = solve(random_graph(32, 0.6, 601))
       c2 = solve(random_graph(64, 0.55, 602))
       return [c1, c2]
   