import random

def construct():
    n = 20
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    
    def is_valid_add(u, v):
        # Check if adding (u, v) creates a K3,3
        # Condition: For all x, y in N(u), |N(v, x, y)| <= 1
        # And for all x, y in N(v), |N(u, x, y)| <= 1
        
        nu = adj[u]
        nv = adj[v]
        
        # Iterate through neighbors of u
        # We need to check pairs x, y in N(u)
        # x, y are indices where bit is set in nu
        # To iterate bits:
        # while mask:
        #   lsb = mask & -mask
        #   i = lsb.bit_length() - 1
        #   mask ^= lsb
        
        # Optimization:
        # If degree is high, this loop is expensive.
        # But n=20 is small.
        
        # Check condition 1
        mask = nu
        while mask:
            lsb_x = mask & -mask
            x = lsb_x.bit_length() - 1
            mask ^= lsb_x
            
            mask_y = mask
            while mask_y:
                lsb_y = mask_y & -mask_y
                y = lsb_y.bit_length() - 1
                mask_y ^= lsb_y
                
                # Check |N(v, x, y)|
                # N(v, x, y) = adj[v] & adj[x] & adj[y]
                # Note: adj[v] does not contain u yet.
                # We are checking the current state.
                common = adj[v] & adj[x] & adj[y]
                if common.bit_count() > 1:
                    return False
        
        # Check condition 2
        mask = nv
        while mask:
            lsb_x = mask & -mask
            x = lsb_x.bit_length() - 1
            mask ^= lsb_x
            
            mask_y = mask
            while mask_y:
                lsb_y = mask_y & -mask_y
                y = lsb_y.bit_length() - 1
                mask_y ^= lsb_y
                
                # Check |N(u, x, y)|
                common = adj[u] & adj[x] & adj[y]
                if common.bit_count() > 1:
                    return False
                    
        return True

    # Greedy construction
    # Try multiple random orders to find a good one
    best_edges = 0
    best_adj = [0]*n
    
    for _ in range(50): # 50 restarts
        adj = [0] * n
        edges = []
        pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    res = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            mask ^= lsb
            if u < v:
                res.append([u, v])
                
    return res
