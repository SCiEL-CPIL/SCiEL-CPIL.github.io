def construct():
    from math import gcd
    import time
    import sys
    
    # Increase recursion limit for deep B&B
    sys.setrecursionlimit(10000)

    n = 256
    candidates = list(range(2, n + 1))
    num_vertices = len(candidates)
    idx_map = {x: i for i, x in enumerate(candidates)}
    
    # Weights
    weights = [1.0 / x for x in candidates]
    
    # Omega parity
    def omega_parity(x):
        count = 0
        d = 2
        while d * d <= x:
            while x % d == 0:
                count += 1
                x //= d
            d += 1
        if x > 1:
            count += 1
        return count % 2

    omegas = [omega_parity(x) for x in candidates]

    # Build conflict graph using bitmasks
    # mask[i] has bit j set if candidates[i] and candidates[j] conflict
    masks = [0] * num_vertices
    
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            a, b = candidates[i], candidates[j]
            g = gcd(a, b)
            if (a // g) * b <= n:
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)

    # Helper to count bits
    def popcount(x):
        return bin(x).count("1")

    # Global best
    best_weight = 0.0
    best_set_mask = 0

    # Upper bound function: Sum of weights of available vertices
    # To make it tighter, we could use a greedy independent set bound, 
    # but sum of weights is a valid upper bound (loose).
    # A better bound: Greedy Max Weight Independent Set on the available subgraph.
    
    def greedy_bound(avail_mask):
        # Greedy: pick vertex with max weight, remove it and neighbors
        # This is O(V^2) which might be slow if called too often.
        # Let's use a simple sum bound first, and maybe improve if needed.
        # Actually, for 256 nodes, sum of weights is very loose.
        # Let's try a fast greedy bound.
        total = 0.0
        curr = avail_mask
        # Sort indices by weight desc? We can pre-sort indices by weight.
        # But avail_mask changes.
        # We can iterate through all vertices and check if they are in avail_mask.
        # To speed up, we can maintain a list of vertices sorted by weight.
        # For now, simple sum is O(V). Greedy is O(V^2).
        # Let's stick to sum for now, it's safe.
        # Wait, sum of weights of ALL remaining is an upper bound.
        # Is it tight enough?
        # Max possible sum is sum(1/x for x in [2,256]) ~ 1.5.
        # The optimal independent set sum is likely close to this if the graph is sparse.
        # If the graph is dense, the bound is bad.
        # The conflict graph is sparse for large numbers.
        # So the bound should be okay.
        
        # Better: Precompute prefix sums of sorted weights? No, avail_mask is arbitrary.
        # Let's use the sum of weights of available vertices.
        s = 0.0
        # Iterate through set bits
        m = avail_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    def solve(avail_mask, current_weight, current_set_mask):
        nonlocal best_weight, best_set_mask
        
        # Prune
        if current_weight + greedy_bound(avail_mask) <= best_weight:
            return
            
        if avail_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set_mask = current_set_mask
            return

        # Reductions
        # 1. Isolated vertices
        changed = True
        while changed:
            changed = False
            m = avail_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                # Check if isolated in avail_mask
                if (masks[idx] & avail_mask) == 0:
                    # Add to solution
                    current_weight += weights[idx]
                    current_set_mask |= lsb
                    avail_mask ^= lsb
                    changed = True
                    # Break and restart loop with new avail_mask
                    break
                m ^= lsb
            if not changed:
                break

        if avail_mask == 0:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set_mask = current_set_mask
            return

        # Prune again after reductions
        if current_weight + greedy_bound(avail_mask) <= best_weight:
            return

        # 2. Domination
        # Find u, v in avail_mask such that N[u] subset N[v] and w[v] >= w[u]
        # This is O(V^2).
        # We can do this once per node or periodically.
        # Let's do a simple pass.
        dominated = [False] * num_vertices
        m = avail_mask
        indices = []
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            indices.append(idx)
            m ^= lsb
            
        for i_idx, u in enumerate(indices):
            if dominated[u]:
                continue
            for j_idx, v in enumerate(indices):
                if i_idx == j_idx or dominated[v]:
                    continue
                # Check if u is dominated by v
                # N[u] subset N[v]
                # N[u] = (masks[u] & avail_mask) | (1<<u)
                # N[v] = (masks[v] & avail_mask) | (1<<v)
                nu = (masks[u] & avail_mask) | (1 << u)
                nv = (masks[v] & avail_mask) | (1 << v)
                if (nu & ~nv) == 0:
                    if weights[v] >= weights[u]:
                        dominated[u] = True
                        break

        if any(dominated):
            new_avail = avail_mask
            for u in indices:
                if dominated[u]:
                    new_avail &= ~(1 << u)
            if new_avail != avail_mask:
                solve(new_avail, current_weight, current_set_mask)
                return # Only one branch needed if we just removed dominated vertices? 
                # Yes, removing dominated vertices doesn't change the optimal solution value.

        # Branching
        # Select vertex to branch on
        # Heuristic: Max degree in available graph
        best_v = -1
        max_deg = -1
        m = avail_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            deg = popcount(masks[idx] & avail_mask)
            # Tie break: Prefer even Omega?
            if deg > max_deg or (deg == max_deg and omegas[idx] == 0):
                max_deg = deg
                best_v = idx
            m ^= lsb
            
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include best_v
        # Remove best_v and its neighbors
        new_avail_include = avail_mask & ~(masks[best_v] | v_bit)
        solve(new_avail_include, current_weight + weights[best_v], current_set_mask | v_bit)
        
        # Branch 2: Exclude best_v
        new_avail_exclude = avail_mask & ~v_bit
        solve(new_avail_exclude, current_weight, current_set_mask)

    # Initial call
    initial_avail = (1 << num_vertices) - 1
    solve(initial_avail, 0.0, 0)

    # Extract result
    result = []
    m = best_set_mask
    while m:
        lsb = m & (-m)
        idx = lsb.bit_length() - 1
        result.append(candidates[idx])
        m ^= lsb
        
    return result
