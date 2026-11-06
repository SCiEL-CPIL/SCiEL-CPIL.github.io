"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n = 40
    target_edges = 401
    
    # Start with K20,20 + 1 intra-edge
    edge_set = set()
    for a in range(20):
        for b in range(20, 40):
            edge_set.add((a, b))
    edge_set.add((0, 1))
    
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edge_set:
        adj[u, v] = True
        adj[v, u] = True
    
    def book_size(u, v):
        c = np.logical_and(adj[u], adj[v])
        c[u] = False
        c[v] = False
        return int(c.sum())
    
    def max_book():
        mx = 0
        for u, v in edge_set:
            b = book_size(u, v)
            if b > mx:
                mx = b
        return mx
    
    current_max = max_book()
    best_edge_set = set(edge_set)
    best_max = current_max
    
    random.seed(42)
    all_non_edges = [(a, b) for a in range(n) for b in range(a+1, n) if (a, b) not in edge_set]
    
    for it in range(100000):
        if time.time() > deadline:
            break
        
        # Find max-book edge
        max_edge = None
        max_b = -1
        for u, v in edge_set:
            b = book_size(u, v)
            if b > max_b:
                max_b = b
                max_edge = (u, v)
        
        if max_edge is None or max_b <= 7:
            break
        
        u, v = max_edge
        improved = False
        
        # Try swapping (u,v) with a random non-edge
        for _ in range(10):
            if not all_non_edges:
                break
            a, b = random.choice(all_non_edges)
            
            # Temporarily remove (u,v), add (a,b)
            adj[u, v] = False
            adj[v, u] = False
            adj[a, b] = True
            adj[b, a] = True
            edge_set.discard((u, v))
            edge_set.add((a, b))
            
            new_max = max_book()
            if new_max < current_max:
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_edge_set = set(edge_set)
                improved = True
                break
            else:
                # Revert
                adj[u, v] = True
                adj[v, u] = True
                adj[a, b] = False
                adj[b, a] = False
                edge_set.discard((a, b))
                edge_set.add((u, v))
        
        if not improved:
            # Restore
            adj[u, v] = True
            adj[v, u] = True
    
    if len(best_edge_set) != target_edges:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_edge_set]
# EVOLVE-BLOCK-END
