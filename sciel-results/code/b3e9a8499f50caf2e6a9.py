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
    Uses Most Constrained Variable (MCV) greedy heuristic.
    Returns the result dict if successful, else None.
    """
    start_time = time.time()
    timeout = 115  # Leave buffer for 120s limit
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # colors[i][j] = color of edge (i,j), -1 if unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # adj[c][v] is a bitmask of neighbors of v in color c
        adj = [[0] * n for _ in range(num_colors)]
        
        # common_count[c][a][b] = number of common neighbors of a and b in color c
        common_count = [[[0]*n for _ in range(n)] for _ in range(num_colors)]
        
        # List of uncolored edges
        uncolored = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        rng = random.Random(attempt * 7919 + 104729)
        
        valid = True
        while uncolored:
            if time.time() - start_time > timeout:
                valid = False
                break
            
            # Find the most constrained edge
            # For each uncolored edge, check which colors are feasible.
            # Feasible: does not create a C4.
            # Cost: number of new at-risk pairs.
            
            best_edge = None
            best_color = -1
            best_feasible_count = float('inf')
            best_cost = float('inf')
            
            # To make it faster, we can break early if we find an edge with 1 feasible color?
            # No, we need to find the global minimum feasible count.
            
            for (u, v) in uncolored:
                feasible_colors = []
                min_cost_for_edge = float('inf')
                
                for c in range(num_colors):
                    cost, feasible = _evaluate_color_bitmask(adj, common_count, u, v, c)
                    if feasible:
                        feasible_colors.append(c)
                        if cost < min_cost_for_edge:
                            min_cost_for_edge = cost
                
                if len(feasible_colors) == 0:
                    valid = False
                    break
                
                # Heuristic: Prefer edges with fewer feasible colors.
                # If tie, prefer edge with lower min cost.
                if len(feasible_colors) < best_feasible_count or \
                   (len(feasible_colors) == best_feasible_count and min_cost_for_edge < best_cost):
                    best_feasible_count = len(feasible_colors)
                    best_cost = min_cost_for_edge
                    best_edge = (u, v)
                    # Pick the best color for this edge
                    # If multiple colors have min_cost, pick one randomly or first?
                    # Let's pick the one with min cost. If tie, random.
                    candidates = [c for c in feasible_colors 
                                  if _evaluate_color_bitmask(adj, common_count, u, v, c)[0] == min_cost_for_edge]
                    best_color = rng.choice(candidates)
            
            if not valid:
                break
            
            if best_edge is None:
                valid = False
                break
            
            u, v = best_edge
            c = best_color
            
            # Assign color c to edge (u, v)
            colors[u][v] = c
            colors[v][u] = c
            
            # Remove from uncolored
            # This is O(N) to remove. N=406. 406*406 = 160k ops. Acceptable.
            # Optimization: Use a set or just filter.
            uncolored = [(a, b) for (a, b) in uncolored if (a != u or b != v)]
            
            # Update adjacency bitmasks
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
            # Update common_count for affected pairs
            # 1. Pair (u, v)
            common_uv = (adj[c][u] & adj[c][v]).bit_count()
            common_count[c][u][v] = common_uv
            common_count[c][v][u] = common_uv
            
            # 2. Pairs (v, w) for w in adj[c][u] (excluding v)
            mask_u = adj[c][u]
            w_mask = mask_u
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                if w == v:
                    continue
                common_vw = (adj[c][v] & adj[c][w]).bit_count()
                common_count[c][v][w] = common_vw
                common_count[c][w][v] = common_vw
                
            # 3. Pairs (u, w) for w in adj[c][v] (excluding u)
            mask_v = adj[c][v]
            w_mask = mask_v
            while w_mask:
                lsb = w_mask & -w_mask
                w = lsb.bit_length() - 1
                w_mask ^= lsb
                if w == u:
                    continue
                common_uw = (adj[c][u] & adj[c][w]).bit_count()
                common_count[c][u][w] = common_uw
                common_count[c][w][u] = common_uw
        
        if valid:
            # Verify no C4 exists
            is_valid = True
            for c in range(num_colors):
                for a in range(n):
                    for b in range(a+1, n):
                        if common_count[c][a][b] >= 2:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                flat_colors = [colors[i][j] for i in range(n) for j in range(i+1, n)]
                return {'n': n, 'colors': flat_colors}
    
    return None

def _evaluate_color_bitmask(adj, common_count, u, v, c):
    """
    Evaluate the cost of assigning color c to edge (u, v).
    Returns (cost, feasible) where:
      - feasible is False if this would create a C4
      - cost is the number of new "at-risk" pairs (pairs going from 0 to 1 common neighbors)
    """
    # 1. Check pair (u, v)
    common_uv = (adj[c][u] & adj[c][v]).bit_count()
    
    if common_uv >= 2:
        return (float('inf'), False)
    
    # 2. Check pairs (v, w) for w in adj[c][u] (w != v)
    cost = 0
    mask_u = adj[c][u]
    while mask_u:
        lsb = mask_u & -mask_u
        w = lsb.bit_length() - 1
        mask_u ^= lsb
        if w == v:
            continue
        if common_count[c][v][w] >= 1:
            return (float('inf'), False)
        else:
            cost += 1
            
    # 3. Check pairs (u, w) for w in adj[c][v] (w != u)
    mask_v = adj[c][v]
    while mask_v:
        lsb = mask_v & -mask_v
        w = lsb.bit_length() - 1
        mask_v ^= lsb
        if w == u:
            continue
        if common_count[c][u][w] >= 1:
            return (float('inf'), False)
        else:
            cost += 1
            
    return (cost, True)
