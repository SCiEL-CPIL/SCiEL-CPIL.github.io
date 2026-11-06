import time
import random
import math

def construct():
    n = 20
    deadline = time.time() + 110  # Leave margin for 120s limit
    
    # Helper to get bit count using Python 3.10+ method
    def popcount(x):
        return x.bit_count()
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    
    # Start with a random greedy construction
    def greedy_construct():
        adj = [0] * n
        edges = []
        edge_list = all_edges[:]
        random.shuffle(edge_list)
        
        for u, v in edge_list:
            # Check 1: |N(u) ∩ N(v)| <= 2
            if popcount(adj[u] & adj[v]) > 2:
                continue
            
            # Check 2: For all w in N(u), |N(w) ∩ N(v)| <= 1
            mask_u = adj[u]
            valid = True
            i = 0
            while mask_u:
                if mask_u & 1:
                    w = i
                    if popcount(adj[w] & adj[v]) > 1:
                        valid = False
                        break
                mask_u >>= 1
                i += 1
            if not valid:
                continue
            
            # Check 3: For all w in N(v), |N(w) ∩ N(u)| <= 1
            mask_v = adj[v]
            i = 0
            while mask_v:
                if mask_v & 1:
                    w = i
                    if popcount(adj[w] & adj[u]) > 1:
                        valid = False
                        break
                mask_v >>= 1
                i += 1
            if not valid:
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append((u, v))
        
        return adj, edges
    
    # Check if edge (u,v) can be added to adj without creating K2,3
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True
        
        if popcount(adj[u] & adj[v]) > 2:
            return False
        
        mask_u = adj[u]
        i = 0
        while mask_u:
            if mask_u & 1:
                w = i
                if popcount(adj[w] & adj[v]) > 1:
                    return False
            mask_u >>= 1
            i += 1
        
        mask_v = adj[v]
        i = 0
        while mask_v:
            if mask_v & 1:
                w = i
                if popcount(adj[w] & adj[u]) > 1:
                    return False
            mask_v >>= 1
            i += 1
        
        return True
    
    # Remove edge (u,v) from adj
    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    # Add edge (u,v) to adj
    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Local search: try to add missing edges
    def local_search_add(adj, edges):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            
            for u, v in edge_list:
                if adj[u] & (1 << v):
                    continue
                
                if can_add_edge(adj, u, v):
                    add_edge(adj, u, v)
                    edges.append((u, v))
                    improved = True
        
        return adj, edges
    
    # Edge swap local search: remove one edge, try to add two
    def local_search_swap(adj, edges):
        improved = True
        max_swaps = 500
        while improved and time.time() < deadline and max_swaps > 0:
            improved = False
            max_swaps -= 1
            
            if not edges:
                break
            
            # Pick a random edge to remove
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            remove_edge(adj, u, v)
            edges.pop(idx)
            
            # Try to add two edges
            added = 0
            edge_list = all_edges[:]
            random.shuffle(edge_list)
            
            for a, b in edge_list:
                if added >= 2:
                    break
                if adj[a] & (1 << b):
                    continue
                if can_add_edge(adj, a, b):
                    add_edge(adj, a, b)
                    edges.append((a, b))
                    added += 1
            
            # If we couldn't add at least one edge, restore
            if added == 0:
                add_edge(adj, u, v)
                edges.append((u, v))
                max_swaps = 0  # Stop swapping if no progress
            
            if added > 0:
                improved = True
        
        return adj, edges
    
    # Full local search combining add and swap
    def full_local_search(adj, edges):
        adj, edges = local_search_add(adj, edges)
        adj, edges = local_search_swap(adj, edges)
        adj, edges = local_search_add(adj, edges)
        return adj, edges
    
    # Main loop
    best_edges = []
    best_count = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        adj, edges = greedy_construct()
        adj, edges = full_local_search(adj, edges)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
        
        # If we have a good solution, do additional polish
        if best_count >= 100 and time.time() < deadline - 20:
            adj_best = [0] * n
            for u, v in best_edges:
                adj_best[u] |= (1 << v)
                adj_best[v] |= (1 << u)
            
            adj_best, edges_best = full_local_search(adj_best, best_edges[:])
            
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
    
    # Final polish pass with more aggressive search
    if best_count > 0 and time.time() < deadline - 5:
        adj_best = [0] * n
        for u, v in best_edges:
            adj_best[u] |= (1 << v)
            adj_best[v] |= (1 << u)
        
        # Multiple rounds of add/swap
        for _ in range(3):
            if time.time() >= deadline - 3:
                break
            adj_best, edges_best = local_search_add(adj_best, best_edges[:])
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
            
            adj_best, edges_best = local_search_swap(adj_best, best_edges[:])
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
        
        # Final add pass
        if time.time() < deadline - 2:
            adj_best = [0] * n
            for u, v in best_edges:
                adj_best[u] |= (1 << v)
                adj_best[v] |= (1 << u)
            adj_best, edges_best = local_search_add(adj_best, best_edges[:])
            if len(edges_best) > best_count:
                best_count = len(edges_best)
                best_edges = edges_best[:]
    
    # Convert list of tuples to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
