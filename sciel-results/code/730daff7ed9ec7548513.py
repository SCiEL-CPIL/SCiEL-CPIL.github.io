import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 115
    n = 16

    # ... find_q3_edges helper ...

    # 1. Constructive Phase
    adj = [0] * n
    edges_order = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges_order)
    
    for u, v in edges_order:
        # Check if adding (u,v) creates a Q3
        # Temporarily add
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        q3 = find_q3_edges(adj)
        if q3 is not None:
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    
    # 2. Local Search Phase (Edge Swapping)
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        
        # Get list of missing edges
        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() - start_time > time_limit:
                break
            
            # Try adding (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            q3_edges = find_q3_edges(adj)
            
            if q3_edges is None:
                # Valid, keep it
                improved = True
                continue
            else:
                # Invalid, try to swap
                # q3_edges is a list of 12 edges
                # Try removing each edge in q3_edges (except u-v)
                swapped = False
                for rem_u, rem_v in q3_edges:
                    if rem_u == u and rem_v == v:
                        continue
                    if rem_u == v and rem_v == u:
                        continue
                    
                    # Tentatively remove (rem_u, rem_v)
                    adj[rem_u] &= ~(1 << rem_v)
                    adj[rem_v] &= ~(1 << rem_u)
                    
                    # Check if valid now
                    q3_check = find_q3_edges(adj)
                    
                    if q3_check is None:
                        # Valid swap
                        improved = True
                        swapped = True
                        break
                    else:
                        # Revert removal
                        adj[rem_u] |= (1 << rem_v)
                        adj[rem_v] |= (1 << rem_u)
                
                if not swapped:
                    # Revert addition of (u, v)
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
    
    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
