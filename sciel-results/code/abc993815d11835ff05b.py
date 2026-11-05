import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def is_c4_free_fast(n, adj):
        """
        Check if graph is C4-free.
        Condition: For all u, v, |N(u) ∩ N(v)| <= 1.
        """
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 1:
                    return False
        return True

    def min_degree_fast(n, adj):
        return min(len(adj[i]) for i in range(n))

    def try_build(n, min_deg_target, time_limit=10.0):
        """
        Improved greedy construction with local search.
        Uses a smarter edge ordering and local search to improve min degree.
        """
        try_start = time.time()
        
        # Generate all potential edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        best_adj = None
        best_min_deg = -1
        
        for attempt in range(50):
            if time.time() > try_start + time_limit:
                break
            
            adj = [set() for _ in range(n)]
            random.shuffle(all_edges)
            
            # Greedy pass: add edges that don't violate C4-free
            for u, v in all_edges:
                if time.time() > try_start + time_limit:
                    break
                
                # Check if adding (u, v) maintains C4-free property
                # Condition 1: |N(u) ∩ N(v)| <= 1
                if len(adj[u] & adj[v]) >= 2:
                    continue
                
                # Condition 2: For all w in N(v), |N(u) ∩ N(w)| == 0 (since v will be added to N(u))
                valid = True
                for w in adj[v]:
                    if adj[u] & adj[w]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Condition 3: For all w in N(u), |N(v) ∩ N(w)| == 0 (since u will be added to N(v))
                for w in adj[u]:
                    if adj[v] & adj[w]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If we pass, add the edge
                adj[u].add(v)
                adj[v].add(u)
            
            # Local search: try to improve min degree by swapping edges
            current_min_deg = min(len(adj[i]) for i in range(n))
            
            # Identify low-degree vertices
            low_deg_verts = [i for i in range(n) if len(adj[i]) < min_deg_target]
            
            # Try to add edges to low-degree vertices
            for _ in range(100):
                if time.time() > try_start + time_limit:
                    break
                
                if not low_deg_verts:
                    break
                
                # Pick a random low-degree vertex
                u = random.choice(low_deg_verts)
                
                # Find vertices not adjacent to u
                non_neighbors = [v for v in range(n) if v != u and v not in adj[u]]
                
                if not non_neighbors:
                    break
                
                random.shuffle(non_neighbors)
                
                # Try to add an edge from u to a non-neighbor
                added = False
                for v in non_neighbors:
                    # Check if adding (u, v) is valid
                    if len(adj[u] & adj[v]) >= 2:
                        continue
                    
                    valid = True
                    for w in adj[v]:
                        if adj[u] & adj[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for w in adj[u]:
                        if adj[v] & adj[w]:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Add the edge
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
                
                # Update low-degree vertices
                low_deg_verts = [i for i in range(n) if len(adj[i]) < min_deg_target]
            
            # Check if we improved
            current_min_deg = min(len(adj[i]) for i in range(n))
            if current_min_deg > best_min_deg:
                best_min_deg = current_min_deg
                best_adj = [set(s) for s in adj]
            
            # If we met the target, return immediately
            if best_min_deg >= min_deg_target:
                edges = []
                for u in range(n):
                    for v in best_adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        
        # Return best found if it meets target
        if best_min_deg >= min_deg_target:
            edges = []
            for u in range(n):
                for v in best_adj[u]:
                    if u < v:
                        edges.append([u, v])
            return edges
        
        return None

    best_result = None
    
    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 5:
            break
        
        min_deg_target = n - 39
        
        # Try multiple times with different random seeds
        found = False
        for attempt in range(15):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 12345 + n)
            edges = try_build(n, min_deg_target, time_limit=8.0)
            
            if edges is not None:
                # Final verification
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                if is_c4_free_fast(n, adj) and min_degree_fast(n, adj) >= min_deg_target:
                    best_result = {'n': n, 'edges': edges}
                    found = True
                    break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: Return a trivial valid graph
        best_result = {'n': 39, 'edges': []}
        
    return best_result
