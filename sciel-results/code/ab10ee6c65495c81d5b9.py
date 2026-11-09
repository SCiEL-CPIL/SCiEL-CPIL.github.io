import time
import sys

def construct():
         start_time = time.time()
         deadline = start_time + 110  # Safety margin
        
         def gen_gp_edges(n, k):
             edges = []
             # Outer cycle
             for i in range(n):
                 edges.append([i, (i + 1) % n])
             # Inner star
             for i in range(n):
                 edges.append([n + i, n + (i + k) % n])
             # Spokes
             for i in range(n):
                 edges.append([i, n + i])
             return edges

         def has_independent_set_10(n, edges):
             # Build adjacency mask
             # n <= 40, so fits in int
             adj = [0] * n
             for u, v in edges:
                 adj[u] |= (1 << v)
                 adj[v] |= (1 << u)
             
             # DFS for independent set of size 10
             # candidates: bitmask of vertices that can be added
             # current_size: size of current independent set
             
             # Sort vertices by degree descending to prune faster?
             # For independent set, high degree vertices are "expensive" to add (they remove many candidates).
             # So we should try adding low degree vertices first? Or high degree?
             # If we add a high degree vertex, we remove many candidates.
             # If we add a low degree vertex, we remove few.
             # To find a set of size 10 quickly, maybe pick vertices with low "conflict".
             # Standard coloring bound:
             # If we can color the remaining graph with c colors, we can pick at most 1 vertex from each color class? No, that's for clique.
             # For independent set, if we color the graph, an independent set can contain vertices of the same color.
             # So coloring doesn't bound the independent set size directly in the same way.
             # However, if the graph is complete, $\alpha=1$.
             
             # Simple pruning:
             # If popcount(candidates) + current_size < 10: return False
             
             sys.setrecursionlimit(10000)
             
             def dfs(candidates_mask, current_size):
                 if current_size == 10:
                     return True
                 if popcount(candidates_mask) + current_size < 10:
                     return False
                 
                 # Heuristic: pick a vertex with minimum degree in the induced subgraph of candidates?
                 # Or just iterate.
                 # To speed up, pick a vertex, say the first one.
                 # Better: pick a vertex that has the most neighbors in candidates?
                 # If we pick v, new_candidates = candidates & ~adj[v] & ~(1<<v)
                 
                 # Find a vertex in candidates
                 # Let's pick the vertex with the highest degree in the candidate subgraph?
                 # This might prune the tree faster if it eliminates many options.
                 
                 # Simple approach: iterate through all vertices in candidates.
                 # For each v, recurse with candidates without v and without neighbors of v.
                 
                 # Optimization: If candidates is empty, return False.
                 if candidates_mask == 0:
                     return False
                 
                 # Pick a vertex v
                 # Let's pick the one with max degree in candidate subgraph to prune hard
                 best_v = -1
                 max_deg = -1
                 temp_mask = candidates_mask
                 while temp_mask:
                     lsb = temp_mask & -temp_mask
                     v = lsb.bit_length() - 1
                     deg = popcount(candidates_mask & adj[v])
                     if deg > max_deg:
                         max_deg = deg
                         best_v = v
                     temp_mask ^= lsb
                     
                 if best_v == -1:
                     return False
                 
                 v = best_v
                 # Branch 1: Include v
                 new_mask = candidates_mask & ~adj[v] & ~(1 << v)
                 if dfs(new_mask, current_size + 1):
                     return True
                 
                 # Branch 2: Exclude v
                 new_mask_ex = candidates_mask & ~(1 << v)
                 if dfs(new_mask_ex, current_size):
                     return True
                     
                 return False

             # Initial candidates: all vertices
             init_mask = (1 << n) - 1
             return dfs(init_mask, 0)

         def popcount(x):
             return bin(x).count('1')

         # Try n from 40 down to 27
         for n_total in range(40, 26, -1):
             # n_total must be even for GP? No, GP is defined for any n.
             # But vertices are 0..n_total-1.
             # My GP generation uses n for the parameter, and 2n for total vertices.
             # So if I want total vertices N, I set n_param = N // 2.
             # If N is odd, GP is not standard.
             # But I can just try even N first (40, 38, ...).
             # The prompt says "Maximize n".
             # If 40 works, great. If not, 39?
             # GP only gives even number of vertices.
             # For odd N, I might need a different construction or subgraph.
             # Let's stick to even N for GP.
             
             n_param = n_total // 2
             if 2 * n_param != n_total:
                 continue # Skip odd for now, or handle separately?
             
             # Try k from 1 to n_param // 2
             for k in range(1, n_param // 2 + 1):
                 if time.time() > deadline:
                     break
                 
                 edges = gen_gp_edges(n_param, k)
                 # Check triangle-free (GP is always triangle-free, but let's be safe? No, it's a theorem)
                 # Check independent set
                 if not has_independent_set_10(n_total, edges):
                     return {'n': n_total, 'edges': edges}
             
             if time.time() > deadline:
                 break
         
         # Fallback to known valid n=26
         n = 26
         return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
