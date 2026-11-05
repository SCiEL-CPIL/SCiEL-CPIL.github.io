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
        Greedy construction using the fast C4 check.
        We add edges if they don't violate the C4-free condition.
        To maximize degree, we prioritize edges that help low-degree vertices.
        """
        try_start = time.time()
        adj = [set() for _ in range(n)]
        
        # Generate all potential edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        # We will do multiple passes or a single randomized pass.
        # A single pass might not be enough to reach high min-degree.
        # Let's do a randomized greedy pass, but repeat if we have time.
        
        for attempt in range(100):
            if time.time() > try_start + time_limit:
                break
            
            adj = [set() for _ in range(n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > try_start + time_limit:
                    break
                
                # Check if adding (u, v) maintains C4-free property.
                # Condition 1: The pair (u, v) itself must satisfy |N(u) ∩ N(v)| <= 1.
                # Since u and v are not connected yet, N(u) and N(v) do not contain each other.
                # If we add (u,v), N(u) gains v, N(v) gains u.
                # The new common neighbors of u and v are the old common neighbors.
                # So we need |N(u) ∩ N(v)| <= 1.
                if len(adj[u] & adj[v]) >= 2:
                    continue
                
                # Condition 2: Adding v to N(u) must not cause any pair (u, w) to have > 1 common neighbor.
                # This happens if v is in N(w) and there is already another common neighbor between u and w.
                # i.e., v in adj[w] and |adj[u] ∩ adj[w]| >= 1 (excluding v, but v is not in adj[u] yet).
                # So if v in adj[w], we need |adj[u] ∩ adj[w]| == 0?
                # If |adj[u] ∩ adj[w]| >= 1, let x be a common neighbor.
                # After adding (u,v), v is in N(u). If v is also in N(w), then v and x are common neighbors of u and w.
                # So yes, we must ensure that for all w, if v in adj[w], then adj[u] ∩ adj[w] is empty.
                
                valid = True
                # Check for u
                for w in adj[v]:
                    if len(adj[u] & adj[w]) > 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check for v
                for w in adj[u]:
                    if len(adj[v] & adj[w]) > 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If we pass, add the edge
                adj[u].add(v)
                adj[v].add(u)
            
            # Check if we met the degree requirement
            if min(len(adj[i]) for i in range(n)) >= min_deg_target:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return edges
        
        return None

    best_result = None
    
    # Try n=47 first
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 5:
            break
        
        min_deg_target = n - 39
        
        # Try multiple times with different random seeds
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            random.seed(attempt * 12345 + n)
            edges = try_build(n, min_deg_target, time_limit=5.0)
            
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
        # Fallback: Return a trivial valid graph (e.g., n=39, no edges)
        # min_degree >= 39-39=0. Empty graph on 39 vertices is C4-free.
        best_result = {'n': 39, 'edges': []}
        
    return best_result
