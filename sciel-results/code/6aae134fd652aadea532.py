import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
       start_time = time.time()
       deadline = start_time + 50 # Safety margin for 60s limit
       
       n = P['n']
       # Adjacency sets for original graph
       adj = [set() for _ in range(n)]
       for u, v in P['edges']:
           adj[u].add(v)
           adj[v].add(u)
           
       # Set of remaining edges as frozensets of tuples for O(1) lookup and removal
       # Using a set of (min(u,v), max(u,v)) to handle undirected nature
       remaining_edges = set()
       for u, v in P['edges']:
           if u > v: u, v = v, u
           remaining_edges.add((u, v))
           
       bicliques = []
       
       # Helper to check if (u,v) is in remaining
       def is_edge(u, v):
           if u > v: u, v = v, u
           return (u, v) in remaining_edges
           
       # Helper to get neighbors in remaining graph
       # We can just check is_edge on the fly for small N, or maintain adj_rem
       # Maintaining adj_rem is better for performance in loops
       adj_rem = [set() for _ in range(n)]
       for u, v in remaining_edges:
           adj_rem[u].add(v)
           adj_rem[v].add(u)
           
       def remove_edge(u, v):
           if u > v: u, v = v, u
           if (u, v) in remaining_edges:
               remaining_edges.remove((u, v))
               adj_rem[u].remove(v)
               adj_rem[v].remove(u)
               
       while remaining_edges:
           if time.time() > deadline:
               break
               
           best_biclique = None
           best_size = -1
           
           # Sample random edges to start growth
           # For dense graphs, sampling 10-20 edges is good
           # For sparse, maybe more. N=30 is small, so we can sample more.
           samples = list(remaining_edges)
           random.shuffle(samples)
           num_samples = min(50, len(samples))
           
           for _ in range(num_samples):
               if time.time() > deadline:
                   break
               u, v = samples[_]
               L = {u}
               R = {v}
               
               # Grow biclique
               changed = True
               while changed:
                   changed = False
                   # Candidates for L: connected to all in R
                   # Candidates for R: connected to all in L
                   
                   # Intersection of neighbors of all in R
                   # start with all vertices
                   cand_L = set(range(n))
                   cand_L.difference_update(L)
                   cand_L.difference_update(R)
                   for r in R:
                       cand_L.intersection_update(adj_rem[r])
                       
                   cand_R = set(range(n))
                   cand_R.difference_update(L)
                   cand_R.difference_update(R)
                   for l in L:
                       cand_R.intersection_update(adj_rem[l])
                       
                   # Choose side to grow
                   # Heuristic: Grow the side with more candidates to maximize size quickly?
                   # Or grow the side that results in larger biclique area?
                   # Let's try adding all candidates to the side with more candidates.
                   if len(cand_L) > len(cand_R):
                       if cand_L:
                           L.update(cand_L)
                           changed = True
                   else:
                       if cand_R:
                           R.update(cand_R)
                           changed = True
                           
               # Calculate size (number of edges covered)
               size = len(L) * len(R)
               # Ensure all edges actually exist in remaining (should be true by construction)
               # But let's double check if we want to be safe, though construction guarantees it.
               
               if size > best_size:
                   best_size = size
                   best_biclique = (L, R)
                   
           if best_biclique is None:
               # Should not happen if remaining_edges is not empty
               # Fallback: pick any edge
               u, v = next(iter(remaining_edges))
               best_biclique = ({u}, {v})
               
           L, R = best_biclique
           # Convert to sorted lists for consistency, though order doesn't matter for correctness
           L_list = sorted(L)
           R_list = sorted(R)
           
           bicliques.append([L_list, R_list])
           
           # Remove all edges in L x R from remaining
           # Iterate over L and R
           for l in L:
               for r in R:
                   remove_edge(l, r)
                   
       # If we broke out due to time, we might have remaining edges.
       # We must cover ALL edges.
       # If we ran out of time, we need a fast fallback to cover remaining edges.
       # Simplest: 1 edge per biclique.
       if remaining_edges:
           for u, v in remaining_edges:
               if time.time() > deadline:
                   # Absolute last resort, just append single edges
                   # This is valid but poor quality
                   bicliques.append([[u], [v]])
               else:
                   bicliques.append([[u], [v]])
                   # No need to remove, we are iterating the set directly
                   # But wait, iterating a set while modifying is bad, but we are not modifying remaining_edges here, just building bicliques.
                   # Actually, we just need to output them.
                   
       # Ensure types are int and list
       final_bicliques = []
       for b in bicliques:
           L = [int(x) for x in b[0]]
           R = [int(y) for y in b[1]]
           final_bicliques.append([L, R])
           
       return final_bicliques
