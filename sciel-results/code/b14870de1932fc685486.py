def construct():
    import random
    import time
    import numpy as np

    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    idx_to_edge = {i: e for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # Precompute adjacency bitmasks for faster triangle check
    # adj_mask[v] is a bitmask of neighbors of v
    
    def is_triangle_free_mask(adj_masks):
        # Check for triangles
        for u in range(n):
            # For each pair of neighbors of u, check if they are connected
            # This is O(N^3) worst case, but with bitmasks it's fast
            neighbors = adj_masks[u]
            # Iterate through neighbors
            temp = neighbors
            while temp:
                v_bit = temp & -temp
                v = v_bit.bit_length() - 1
                temp -= v_bit
                # Check if any neighbor of v is also a neighbor of u (other than v)
                # common = adj_masks[v] & adj_masks[u]
                # if common has any bits other than u and v, triangle exists?
                # Actually, if (u,v) is edge, and (v,w) is edge, and (w,u) is edge.
                # adj_masks[u] contains v and w.
                # adj_masks[v] contains u and w.
                # Intersection contains u, v, w.
                # We just need to check if adj_masks[u] & adj_masks[v] has any bit set 
                # that corresponds to a vertex w != u and w != v.
                common = adj_masks[u] & adj_masks[v]
                # Remove u and v from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common != 0:
                    return False
        return True

    def is_3_colorable_dsatur(adj_masks):
        # DSATUR heuristic for 3-colorability
        # color[v] = 0, 1, 2 or -1
        color = [-1] * n
        sat = [0] * n # number of different colors in neighbors
        # Order vertices dynamically
        
        def get_saturation(v):
            colors = set()
            for u in range(n):
                if (adj_masks[v] >> u) & 1:
                    if color[u] != -1:
                        colors.add(color[u])
            return len(colors)
            
        # Precompute neighbor lists from masks for easier iteration in DSATUR
        neighbors = [0] * n
        for v in range(n):
            neighbors[v] = [u for u in range(n) if (adj_masks[v] >> u) & 1]
            
        def backtrack(idx):
            if idx == n:
                return True
            
            # Select uncolored vertex with highest saturation, then highest degree
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in range(n):
                if color[v] == -1:
                    s = get_saturation(v)
                    d = len(neighbors[v])
                    if s > best_sat or (s == best_sat and d > best_deg):
                        best_v = v
                        best_sat = s
                        best_deg = d
            
            if best_v == -1:
                return True # All colored
                
            v = best_v
            # Try colors
            # Sort colors by preference? Just 0,1,2
            for c in range(3):
                # Check if c is available
                valid = True
                for u in neighbors[v]:
                    if color[u] == c:
                        valid = False
                        break
                if valid:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
            
        return backtrack(0)

    def get_adj_masks(edge_set):
        masks = [0] * n
        for u, v in edge_set:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    def count_edges(edge_set):
        return len(edge_set)

    best_edges = 0
    best_edge_set = set()
    
    start_time = time.time()
    time_limit = 110
    
    # Initialize with Grötzsch graph
    groetzsch_edges = set()
    for i in range(5):
        groetzsch_edges.add((i, (i+1) % 5))
    for i in range(5):
        u, w = i, (i+1) % 5
        groetzsch_edges.add((u, 5+w))
        groetzsch_edges.add((w, 5+u))
    for i in range(5):
        groetzsch_edges.add((5+i, 10))
        
    if is_triangle_free_mask(get_adj_masks(groetzsch_edges)) and not is_3_colorable_dsatur(get_adj_masks(groetzsch_edges)):
        best_edge_set = groetzsch_edges
        best_edges = len(groetzsch_edges)

    # Simulated Annealing
    random.seed(42)
    
    def evaluate(edge_set):
        # Returns (is_valid_tf, is_3_col, num_edges)
        masks = get_adj_masks(edge_set)
        tf = is_triangle_free_mask(masks)
        if not tf:
            return False, False, 0
        col = is_3_colorable_dsatur(masks)
        return True, col, len(edge_set)
    
    # Start from best known
    current_edges = set(best_edge_set)
    current_masks = get_adj_masks(current_edges)
    current_tf, current_col, current_count = evaluate(current_edges)
    
    T = 100.0
    cooling = 0.999
    
    iterations = 0
    while time.time() - start_time < time_limit:
        iterations += 1
        if T < 0.01:
            T = 100.0
            # Restart from best
            current_edges = set(best_edge_set)
            current_masks = get_adj_masks(current_edges)
            current_tf, current_col, current_count = evaluate(current_edges)
            
        # Generate neighbor
        # Randomly add or remove an edge
        move_type = random.random()
        if move_type < 0.5:
            # Add edge
            candidates = [e for e in all_edges if e not in current_edges]
            if not candidates: continue
            e = random.choice(candidates)
            new_edges = current_edges | {e}
        else:
            # Remove edge
            if not current_edges: continue
            e = random.choice(list(current_edges))
            new_edges = current_edges - {e}
        
        # Check triangle freedom quickly
        # Only check if adding edge
        if move_type < 0.5:
            u, v = e
            if current_masks[u] & current_masks[v]:
                # Triangle formed, skip
                continue
        
        new_masks = get_adj_masks(new_edges)
        new_tf, new_col, new_count = evaluate(new_edges)
        
        if not new_tf:
            continue
            
        # Energy calculation
        # We want to maximize edges, and prefer non-3-colorable
        # Energy = -edges * 10 + (1000 if 3-colorable else 0)
        # Lower is better
        if current_tf:
            E_old = -current_count * 10 + (1000 if current_col else 0)
        else:
            E_old = 1000000 # Invalid state penalty
            
        E_new = -new_count * 10 + (1000 if new_col else 0)
        
        dE = E_new - E_old
        
        if dE < 0 or random.random() < np.exp(-dE / T):
            current_edges = new_edges
            current_masks = new_masks
            current_tf, current_col, current_count = new_tf, new_col, new_count
            
            if not current_col and current_count > best_edges:
                best_edges = current_count
                best_edge_set = set(current_edges)
                
        T *= cooling

    return [list(e) for e in best_edge_set]
