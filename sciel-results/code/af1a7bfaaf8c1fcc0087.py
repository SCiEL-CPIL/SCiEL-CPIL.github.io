import time
import random
from itertools import combinations

def _is_c4_free(n, adj):
    """Check if graph is C4-free using bitsets for speed. adj is list of bitmasks."""
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            common = adj[u] & adj[v]
            # Count bits: if >= 2, C4 exists
            c = common
            if c & (c - 1):  # More than one bit set
                return False
    return True

def _min_degree(n, adj):
    """Compute minimum degree from adjacency bitmasks."""
    min_deg = n
    for u in range(n):
        d = bin(adj[u]).count('1')
        if d < min_deg:
            min_deg = d
    return min_deg

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    best_n = 39
    best_edges = []
    
    # Try to build graphs for n from 47 down to 40
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
        min_deg_needed = target_n - 39
        
        # Strategy: Start with a structured dense graph, then local search
        # For C4-free graphs, we can use polarity graphs or finite field constructions
        
        # Approach: Start with a random graph biased toward high degree,
        # then use local search to fix C4s and boost min degree
        
        found = False
        
        for seed in range(2000):
            if time.time() > deadline:
                break
            
            random.seed(seed)
            
            # Initialize with a moderately dense random graph
            # Start with each edge included with probability p
            # We want min degree ~ target_n - 39, which is small (8-14 for n=46-47)
            # So we can start sparser and add edges
            
            n = target_n
            adj = [0] * n
            deg = [0] * n
            
            # Start with a random graph with moderate density
            # To get min degree ~10 on 47 vertices, average degree should be ~12-15
            # So p ~ 12/(47-1) ~ 0.26
            p = 0.25
            
            # Add random edges
            all_edges = list(combinations(range(n), 2))
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline:
                    break
                if random.random() < p:
                    # Check if adding (u,v) creates a C4
                    creates_c4 = False
                    nv = adj[v]
                    nu = adj[u]
                    
                    w_mask = nv
                    while w_mask:
                        w_bit = w_mask & (-w_mask)
                        w = w_bit.bit_length() - 1
                        w_mask ^= w_bit
                        
                        if w == u:
                            continue
                        
                        x_mask = nu & ~w_bit & ~(1 << v)
                        if x_mask & adj[w]:
                            creates_c4 = True
                            break
                    
                    if not creates_c4:
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                        deg[u] += 1
                        deg[v] += 1
            
            # Now do local search: try to boost min degree while maintaining C4-free
            # Iterate: find vertex with min degree, try to add edges to it
            # If can't add, try to remove an edge from a high-degree vertex and add to low-degree
            
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Check if we've reached target
                min_deg = _min_degree(n, adj)
                if min_deg >= min_deg_needed:
                    # Verify C4-free
                    if _is_c4_free(n, adj):
                        # Convert adj to edges
                        edges = []
                        for u in range(n):
                            for v in range(u + 1, n):
                                if (adj[u] >> v) & 1:
                                    edges.append([u, v])
                        if target_n > best_n:
                            best_n = target_n
                            best_edges = edges
                        found = True
                        break
                
                # Find vertex with minimum degree
                min_deg_vertex = 0
                min_deg_val = n
                for u in range(n):
                    d = bin(adj[u]).count('1')
                    if d < min_deg_val:
                        min_deg_val = d
                        min_deg_vertex = u
                
                # Try to add edges to min_deg_vertex
                added = False
                candidates = [v for v in range(n) if v != min_deg_vertex and not ((adj[min_deg_vertex] >> v) & 1)]
                random.shuffle(candidates)
                
                for v in candidates:
                    if time.time() > deadline:
                        break
                    
                    # Check if adding (min_deg_vertex, v) creates a C4
                    creates_c4 = False
                    nu = adj[min_deg_vertex]
                    nv = adj[v]
                    
                    w_mask = nv
                    while w_mask:
                        w_bit = w_mask & (-w_mask)
                        w = w_bit.bit_length() - 1
                        w_mask ^= w_bit
                        
                        if w == min_deg_vertex:
                            continue
                        
                        x_mask = nu & ~w_bit & ~(1 << v)
                        if x_mask & adj[w]:
                            creates_c4 = True
                            break
                    
                    if not creates_c4:
                        adj[min_deg_vertex] |= 1 << v
                        adj[v] |= 1 << min_deg_vertex
                        deg[min_deg_vertex] += 1
                        deg[v] += 1
                        added = True
                        break
                
                if not added:
                    # Can't add more edges to this vertex without creating C4
                    # Try a different strategy: swap edges
                    # Remove an edge from a high-degree vertex and add to low-degree
                    # This is complex, so just break and try next seed
                    break
            
            if found:
                break
        
        if found:
            break
    
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
