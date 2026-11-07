     def construct():
         from random import Random

         def instance(n, k, p, seed):
             rng = Random(seed)
             edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
             edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
             return sorted(edges)

         results = []
         for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
             n, k, p, seed = spec
             edges = instance(n, k, p, seed)
             
             # Build adjacency for independent set vertices
             adj = {v: [] for v in range(k, n)}
             for u, v in edges:
                 if u < k and v >= k:
                     adj[v].append(u)
             
             # Find if any v in I has degree k
             full_v = -1
             for v in range(k, n):
                 if len(adj[v]) == k:
                     full_v = v
                     break
             
             cliques = []
             
             if full_v != -1:
                 # Use C U {full_v}
                 clique = list(range(k)) + [full_v]
                 cliques.append(clique)
                 # Cover others
                 for v in range(k, n):
                     if v == full_v:
                         continue
                     if len(adj[v]) > 0:
                         clique = adj[v] + [v]
                         cliques.append(clique)
             else:
                 # Use C
                 cliques.append(list(range(k)))
                 # Cover all v in I with deg > 0
                 for v in range(k, n):
                     if len(adj[v]) > 0:
                         clique = adj[v] + [v]
                         cliques.append(clique)
             
             results.append(cliques)
             
         return results
     