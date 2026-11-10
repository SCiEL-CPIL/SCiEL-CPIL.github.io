def construct():
       import time
       from random import Random

       def instance(n, k, p, seed):
           rng = Random(seed)
           edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
           edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
           return sorted(edges)

       def partition_split_graph(n, k, edges):
           cliques = []
           cross_edges = []
           c_c_edges = []
           
           for u, v in edges:
               if u < k and v < k:
                   c_c_edges.append((u, v))
               elif u >= k and v >= k:
                   # Should not happen in split graph with I independent
                   pass
               else:
                   # Cross edge
                   c = u if u < k else v
                   i = v if u < k else u
                   cross_edges.append((c, i))
           
           # 1. Add cliques for cross edges (Singleton C-vertex + I-vertex)
           for c, i in cross_edges:
               cliques.append([c, i])
           
           # 2. Add clique for C-C edges
           # Since C is a clique, all c_c_edges are covered by the set C
           if k >= 2:
               cliques.append(list(range(k)))
           
           return cliques

       results = []
       for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
           n, k, p, seed = spec
           edges = instance(n, k, p, seed)
           cliques = partition_split_graph(n, k, edges)
           cleaned_cliques = [[int(x) for x in c] for c in cliques]
           results.append(cleaned_cliques)
       
       return results
