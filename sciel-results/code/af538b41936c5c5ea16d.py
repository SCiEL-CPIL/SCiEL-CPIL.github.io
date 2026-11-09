import random
import time

def construct():
    """
    Greedy construction with forbidden-pair counting for 5-colourings of K_n
    without monochromatic C4. Maximizes n up to 29.
    Uses bitmask adjacency for speed.
    """
    max_n = 29
    num_colors = 5
    
    # Try to build from largest n down
    for n in range(max_n, 10, -1):
        result = _build_coloring(n, num_colors)
        if result is not None:
            return result
    
    # Fallback to known construction for n=11
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _build_coloring(n, num_colors, max_attempts=5000):
    """
    Attempt to build a valid 5-coloring of K_n without monochromatic C4.
    Uses forbidden-pair counting to guide color assignment.
    Returns the result dict if successful, else None.
    """
    start_time = time.time()
    timeout = 115  # Leave buffer for 120s limit
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # adj[c][v] is a bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(num_colors)]
        colors = [[-1] * n for _ in range(n)]
        
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        rng = random.Random(attempt * 7919 + 104729)
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        valid = True
        for (u, v) in shuffled_edges:
            if time.time() - start_time > timeout:
                valid = False
                break
            
            best_color = -1
            best_cost = float('inf')
            
            for c in range(num_colors):
                cost, feasible = _evaluate_color_fast(adj, u, v, c)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_color = c
            
            if best_color == -1:
                valid = False
                break
            
            c = best_color
            colors[u][v] = c
            colors[v][u] = c
            
            # Update adjacency bitmasks
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
        
        if valid:
            # Verification is implicitly done by the greedy check, 
            # but a final check is cheap and safe.
            # However, the greedy check ensures no C4 is formed at each step.
            # So if we finish, it's valid.
            flat_colors = [colors[i][j] for i in range(n) for j in range(i+1, n)]
            return {'n': n, 'colors': flat_colors}
    
    return None

def _evaluate_color_fast(adj, u, v, c):
    """
    Evaluate the cost of assigning color c to edge (u, v).
    Returns (cost, feasible).
    """
    # 1. Check if u and v already have >= 2 common neighbors in color c
    common_uv = (adj[c][u] & adj[c][v]).bit_count()
    if common_uv >= 2:
        return (float('inf'), False)
    
    cost = 0
    
    # 2. Check neighbors of u in color c
    # For each w in N_c(u), adding (u,v) makes u a common neighbor of v and w?
    # No. (u,v) is the new edge.
    # The new edge (u,v) connects u and v.
    # Any C4 involving (u,v) must have the form (u, x, v, y) or similar.
    # Specifically, if there is a w such that (u,w) is color c and (v,w) is color c,
    # then u and v have a common neighbor w.
    # If they have TWO such common neighbors, C4.
    # We already checked common_uv < 2.
    #
    # What about C4s where (u,v) is a diagonal? No, C4 is a cycle.
    # The condition "no two vertices with two common neighbors" is equivalent to no C4.
    # So we just need to ensure that after adding (u,v), no pair has >= 2 common neighbors.
    #
    # Adding (u,v) affects common neighbor counts for:
    # - Pair (u, w) for w in N_c(v): v becomes a common neighbor of u and w?
    #   No. v is neighbor of u (new) and v is neighbor of w (existing).
    #   So v is a common neighbor of u and w.
    #   We need to check if u and w already have >= 1 common neighbor.
    #   If they do, adding v makes it >= 2.
    # - Pair (v, w) for w in N_c(u): u becomes a common neighbor of v and w.
    #   We need to check if v and w already have >= 1 common neighbor.
    
    # Check neighbors of u
    mask_u = adj[c][u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        mask_u ^= lsb
        
        # We are checking pair (v, w).
        # u is a common neighbor of v and w?
        # (u,v) is being added. (u,w) is existing (since w in N_c(u)).
        # So u is a common neighbor of v and w.
        # Does (v, w) already have a common neighbor?
        # common_vw = (adj[c][v] & adj[c][w]).bit_count()
        # Note: adj[c][v] does NOT yet have u. adj[c][w] has u?
        # adj[c][w] has u because (u,w) is color c.
        # adj[c][v] does not have u yet.
        # So (adj[c][v] & adj[c][w]) does NOT include u.
        # It includes other common neighbors.
        # If count >= 1, then adding u makes it >= 2.
        
        if (adj[c][v] & adj[c][w]).bit_count() >= 1:
            return (float('inf'), False)
        else:
            cost += 1 # Heuristic: cost for creating a "1-common-neighbor" pair
            
    # Check neighbors of v
    mask_v = adj[c][v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        mask_v ^= lsb
        
        # We are checking pair (u, w).
        # v is a common neighbor of u and w?
        # (u,v) is being added. (v,w) is existing.
        # So v is a common neighbor of u and w.
        # Does (u, w) already have a common neighbor?
        # common_uw = (adj[c][u] & adj[c][w]).bit_count()
        # adj[c][u] does NOT have v yet.
        # adj[c][w] has v.
        # So intersection does not include v.
        
        if (adj[c][u] & adj[c][w]).bit_count() >= 1:
            return (float('inf'), False)
        else:
            cost += 1
            
    return (cost, True)
