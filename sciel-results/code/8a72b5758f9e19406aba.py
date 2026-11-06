import time
import random

def construct():
         start_time = time.time()
         deadline = 140.0 # Safety margin
         
         best_n = 39
         best_edges = []
         
         for n in [46, 45, 44, 43]:
             if time.time() > deadline - 10:
                 break
             
             target_deg = n - 39
             # Try multiple restarts
             found = False
             for _ in range(50): # 50 restarts
                 if time.time() > deadline - 5:
                     break
                 
                 edges = set()
                 adj = [set() for _ in range(n)]
                 
                 # Greedy fill
                 # We want to add edges to satisfy min degree
                 # Start with a random sparse graph to avoid immediate C4s?
                 # Or start empty?
                 # Start empty is safer for C4, but harder to reach high degree.
                 # Let's start with a random matching to seed degrees.
                 # Actually, just greedy from empty.
                 
                 # Order vertices by random permutation
                 perm = list(range(n))
                 random.shuffle(perm)
                 
                 # We will try to add edges.
                 # To ensure min degree, we track degrees.
                 deg = [0] * n
                 
                 # Phase 1: Add edges greedily
                 # We iterate until no more edges can be added or time runs out
                 # But we need to prioritize low degree vertices.
                 
                 # Simple loop:
                 # While min(deg) < target_deg:
                 #   v = argmin(deg)
                 #   candidates = [u for u in range(n) if u != v and u not in adj[v]]
                 #   random.shuffle(candidates)
                 #   for u in candidates:
                 #       if is_safe(v, u, adj):
                 #           add edge
                 #           break
                 #   if no edge added:
                 #       break # Stuck
                 
                 # This is too slow if we do it naively.
                 # Optimization:
                 # Precompute common neighbors?
                 # is_safe(v,u) checks if N(v) and N(u) have an edge.
                 
                 # Let's implement is_safe efficiently.
                 
                 def is_safe(v, u):
                     # Check if any a in N(v) is connected to any b in N(u)
                     for a in adj[v]:
                         if adj[a] & adj[u]: # Intersection of neighbors
                             return False
                     return True
                 
                 # Wait, adj[a] & adj[u] is O(1) if sets.
                 # But adj[a] is a set.
                 
                 # Run the greedy loop
                 stuck = False
                 while min(deg) < target_deg:
                     # Pick v with min degree
                     v = min(range(n), key=lambda x: deg[x])
                     
                     # Candidates for u
                     # Sort by degree of u (low first to balance)
                     candidates = [u for u in range(n) if u != v and u not in adj[v]]
                     candidates.sort(key=lambda x: deg[x])
                     
                     added = False
                     for u in candidates:
                         if is_safe(v, u):
                             # Add edge
                             adj[v].add(u)
                             adj[u].add(v)
                             deg[v] += 1
                             deg[u] += 1
                             edges.add((v, u))
                             added = True
                             break
                     
                     if not added:
                         stuck = True
                         break
                 
                 if not stuck and min(deg) >= target_deg:
                     # Verify C4-free
                     # The greedy check ensures no C4 is created.
                     # So it is C4-free.
                     
                     # Convert to list of lists
                     edge_list = [[u, v] for u, v in edges]
                     return {'n': n, 'edges': edge_list}
             
         # Fallback
         return {'n': 39, 'edges': []}
