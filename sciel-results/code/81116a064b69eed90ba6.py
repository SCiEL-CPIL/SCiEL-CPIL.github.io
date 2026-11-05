def construct():
    from math import gcd
    import time
    import random

    n = 256
    candidates = list(range(2, n + 1))
    
    # Precompute conflicts
    conflicts = {x: set() for x in candidates}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            g = gcd(a, b)
            if (a // g) * b <= n:
                conflicts[a].add(b)
                conflicts[b].add(a)

    # Split into Core and Periphery
    # Core: small numbers, high connectivity
    # Periphery: large numbers, low connectivity
    core_limit = 16
    core = [x for x in candidates if x <= core_limit]
    periphery = [x for x in candidates if x > core_limit]
    
    # 1. Generate all independent sets in the Core
    core_sets = []
    
    # Sort core by degree descending for better pruning in enumeration?
    # Not strictly necessary for 15 nodes, but good practice.
    core_sorted = sorted(core, key=lambda x: len(conflicts[x]), reverse=True)
    
    def gen_core_sets_opt(idx, current_set):
        if idx == len(core_sorted):
            core_sets.append(frozenset(current_set))
            return
        
        x = core_sorted[idx]
        # Exclude
        gen_core_sets_opt(idx + 1, current_set)
        
        # Include
        if not (conflicts[x] & current_set):
            gen_core_sets_opt(idx + 1, current_set | {x})

    gen_core_sets_opt(0, frozenset())
    
    # Helper to solve MWIS for a small component (list of nodes)
    def solve_mwis_component(comp_nodes):
        if not comp_nodes:
            return 0.0, set()
        if len(comp_nodes) == 1:
            return 1.0 / comp_nodes[0], {comp_nodes[0]}
        
        # Map nodes to indices
        node_map = {v: i for i, v in enumerate(comp_nodes)}
        k = len(comp_nodes)
        
        # Build adjacency mask
        # adj_mask[i] is a bitmask of neighbors of node i
        adj_mask = [0] * k
        for i in range(k):
            u = comp_nodes[i]
            for v in comp_nodes:
                if u == v: continue
                if v in conflicts[u]:
                    j = node_map[v]
                    adj_mask[i] |= (1 << j)
        
        # DP over bitmasks
        # dp[mask] = (max_weight, set_of_nodes_indices)
        # To save memory/time, we can just store weight and reconstruct later,
        # or store the set. Since k is small, storing set is okay but might be slow.
        # Let's store weight and the decision (include/exclude lowest bit).
        
        # Actually, for k <= 15, 2^15 = 32768 states.
        # We can just use recursion with memoization on mask.
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dp(mask):
            if mask == 0:
                return 0.0, frozenset()
            
            # Get lowest set bit
            lsb = mask & -mask
            i = lsb.bit_length() - 1
            
            # Option 1: Exclude i
            w1, s1 = dp(mask ^ lsb)
            
            # Option 2: Include i
            # Neighbors of i are removed from mask
            new_mask = mask ^ lsb
            new_mask &= ~adj_mask[i]
            w2, s2 = dp(new_mask)
            w2 += 1.0 / comp_nodes[i]
            s2 = s2 | {i}
            
            if w1 > w2:
                return w1, s1
            else:
                return w2, s2
        
        w, s_indices = dp((1 << k) - 1)
        s_nodes = {comp_nodes[i] for i in s_indices}
        return w, s_nodes

    best_sum = 0.0
    best_set = set()
    
    # Pre-sort periphery? Not needed for component decomposition.
    
    for cs in core_sets:
        core_sum = sum(1.0/x for x in cs)
        
        # Filter periphery
        valid_per = [p for p in periphery if not (conflicts[p] & cs)]
        
        if not valid_per:
            total_sum = core_sum
            total_set = set(cs)
        else:
            # Find connected components in the subgraph induced by valid_per
            visited = set()
            components = []
            
            for start in valid_per:
                if start in visited:
                    continue
                # BFS
                comp = []
                queue = [start]
                visited.add(start)
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in conflicts[u]:
                        if v in valid_per and v not in visited:
                            visited.add(v)
                            queue.append(v)
                components.append(comp)
            
            p_sum = 0.0
            p_set = set()
            
            for comp in components:
                w, s = solve_mwis_component(comp)
                p_sum += w
                p_set.update(s)
            
            total_sum = core_sum + p_sum
            total_set = cs | p_set
            
        if total_sum > best_sum:
            best_sum = total_sum
            best_set = total_set
            
    return list(best_set)
