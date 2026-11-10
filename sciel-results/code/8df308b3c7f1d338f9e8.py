import random

def construct():
    n = 20
    best_edges = 0
    best_graph = []
    
    # Precompute all pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Use random shuffling for diverse orderings
    # Fixed seed for reproducibility
    rng = random.Random(42)
    
    # Increase number of restarts significantly
    num_runs = 2000
    
    for _ in range(num_runs):
        current_pairs = pairs[:]
        rng.shuffle(current_pairs)
        
        adj = [0] * n  # Bitsets
        edges = []
        
        for u, v in current_pairs:
            # Check 1: Common neighbors of u and v
            common_uv = adj[u] & adj[v]
            if common_uv.bit_count() > 2:
                continue
            
            valid = True
            
            # Check 2: For all x in N(v), |N(u) & N(x)| <= 1
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                x = lsb.bit_length() - 1
                mask_v ^= lsb
                common_ux = adj[u] & adj[x]
                if common_ux.bit_count() > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 3: For all y in N(u), |N(v) & N(y)| <= 1
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                y = lsb.bit_length() - 1
                mask_u ^= lsb
                common_vy = adj[v] & adj[y]
                if common_vy.bit_count() > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges[:]
            
    return best_graph
