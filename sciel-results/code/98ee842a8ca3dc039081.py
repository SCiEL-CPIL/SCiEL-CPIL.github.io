import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 16
    
    def is_3_colorable(adj_mask):
        """Fast 3-colorability check using backtracking with MRV heuristic and bitmasks."""
        color = [-1] * n
        
        def backtrack():
            # Find uncolored vertex with most colored neighbors (MRV)
            best_v = -1
            best_cnt = -1
            for v in range(n):
                if color[v] == -1:
                    # Count colored neighbors
                    cnt = 0
                    m = adj_mask[v]
                    while m:
                        lsb = m & (-m)
                        u = lsb.bit_length() - 1
                        if color[u] >= 0:
                            cnt += 1
                        m ^= lsb
                    if cnt > best_cnt:
                        best_cnt = cnt
                        best_v = v
            
            if best_v == -1:
                return True
            
            v = best_v
            used = 0
            m = adj_mask[v]
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                if color[u] >= 0:
                    used |= (1 << color[u])
                m ^= lsb
            
            for c in range(3):
                if not (used & (1 << c)):
                    color[v] = c
                    if backtrack():
                        return True
                    color[v] = -1
            return False
        
        return backtrack()

    def edge_count(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2

    def add_edge(adj_mask, u, v):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def remove_edge(adj_mask, u, v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    def is_triangle_free(adj_mask):
        """Check if graph is triangle-free using bitmasks."""
        for u in range(n):
            for v in range(u+1, n):
                if adj_mask[u] & (1 << v):
                    common = adj_mask[u] & adj_mask[v] & ~(1 << u) & ~(1 << v)
                    if common:
                        return False
        return True

    def is_valid(adj_mask):
        """Check triangle-free and not 3-colorable."""
        if not is_triangle_free(adj_mask):
            return False
        return not is_3_colorable(adj_mask)

    # Build initial graph: Grötzsch graph (Mycielski of C5) on vertices 0-10
    adj_mask = [0] * n
    
    # C5: 0-1-2-3-4-0
    for i in range(5):
        add_edge(adj_mask, i, (i+1) % 5)
    
    # Mycielski: add vertices 5-9 as copies of 0-4
    # For each edge (u,v) in C5, add edges (u, 5+v) and (v, 5+u)
    for i in range(5):
        u, v = i, (i+1) % 5
        add_edge(adj_mask, u, 5+v)
        add_edge(adj_mask, v, 5+u)
    
    # Connect 5-i to 10 for i=0..4
    for i in range(5):
        add_edge(adj_mask, 5+i, 10)
    
    # Now we have 11 vertices (0-10). Vertices 11-15 are isolated.
    # Strategy: Add vertices 11-15 by connecting them to independent sets.
    
    def find_independent_sets(adj_mask, vertices, min_size=1):
        """Find all independent sets of size >= min_size among the given vertices."""
        # For n=16, we can just iterate all subsets? 2^16 is 65536. Manageable.
        # But we only care about subsets of `vertices`.
        # Let's use a simple recursive generation or bitmask iteration.
        sets = []
        # We want sets that are independent in the induced subgraph on `vertices`
        # Actually, the set must be independent in the *current* graph.
        
        # Optimization: Just try large sets first.
        # We'll generate all subsets of `vertices` and filter.
        v_list = list(vertices)
        k = len(v_list)
        for mask in range(1 << k):
            if bin(mask).count('1') < min_size:
                continue
            subset = []
            valid = True
            for i in range(k):
                if mask & (1 << i):
                    subset.append(v_list[i])
            
            # Check if subset is independent
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if adj_mask[subset[i]] & (1 << subset[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                sets.append(subset)
        
        # Sort by size descending
        sets.sort(key=lambda x: len(x), reverse=True)
        return sets

    # Add vertices 11-15
    for v_new in range(11, n):
        # Get current vertices
        current_verts = list(range(v_new))
        # Find independent sets in the current graph
        # To speed up, we can limit the search to sets of size >= some threshold
        # Or just try all and pick the largest valid one
        candidates = find_independent_sets(adj_mask, current_verts, min_size=1)
        
        added = False
        for iso_set in candidates:
            # Try adding v_new connected to iso_set
            new_adj = list(adj_mask)
            for u in iso_set:
                add_edge(new_adj, v_new, u)
            
            # Check validity
            if is_valid(new_adj):
                adj_mask = new_adj
                added = True
                break
        
        if not added:
            # If no independent set works, add v_new isolated
            # This might break 4-chromaticity if it was relying on structure, 
            # but adding an isolated vertex doesn't change chromatic number.
            # So if G is 4-chromatic, G+isolated is 4-chromatic.
            # But we want to maximize edges, so we failed to add any edges.
            pass

    # Local Search Phase
    best_adj = list(adj_mask)
    best_count = edge_count(best_adj)
    
    def local_search(adj_mask, max_time):
        current = list(adj_mask)
        current_count = edge_count(current)
        best_local = list(current)
        best_local_count = current_count
        
        no_improve = 0
        while time.time() < max_time and no_improve < 100:
            # Get existing and missing edges
            existing = [(i, j) for i in range(n) for j in range(i+1, n) if current[i] & (1 << j)]
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (current[i] & (1 << j))]
            
            if not missing:
                break
            
            # Try to add a random missing edge
            u, v = random.choice(missing)
            
            # Check triangle-free
            common = current[u] & current[v] & ~(1 << u) & ~(1 << v)
            if not common:
                new_adj = list(current)
                add_edge(new_adj, u, v)
                if not is_3_colorable(new_adj):
                    add_edge(current, u, v)
                    current_count += 1
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local = list(current)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            else:
                # Triangle would form. Try swap.
                # Remove a random existing edge (a,b) that is part of a triangle with (u,v)?
                # Or just remove a random existing edge.
                if existing:
                    a, b = random.choice(existing)
                    remove_edge(current, a, b)
                    current_count -= 1
                    
                    # Now try to add (u,v) again
                    common2 = current[u] & current[v] & ~(1 << u) & ~(1 << v)
                    if not common2:
                        new_adj = list(current)
                        add_edge(new_adj, u, v)
                        if not is_3_colorable(new_adj):
                            add_edge(current, u, v)
                            current_count += 1
                            # Net change 0, but maybe better structure?
                            # For now, only accept if count increases or same but we track best.
                            if current_count > best_local_count:
                                best_local_count = current_count
                                best_local = list(current)
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            # Revert
                            remove_edge(current, u, v) # It wasn't added
                            add_edge(current, a, b)
                            current_count += 1
                            no_improve += 1
                    else:
                        # Still triangle, revert
                        add_edge(current, a, b)
                        current_count += 1
                        no_improve += 1
                else:
                    no_improve += 1
            
            # Random restarts/perturbations if stuck
            if no_improve > 20:
                # Remove a random edge
                if existing:
                    a, b = random.choice(existing)
                    remove_edge(current, a, b)
                    current_count -= 1
                no_improve = 0
        
        return best_local, best_local_count

    # Run local search
    adj_mask, local_count = local_search(adj_mask, deadline - 10)
    if local_count > best_count:
        best_count = local_count
        best_adj = list(adj_mask)
    
    # Multi-start
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        # Perturb best_adj
        current = list(best_adj)
        existing = [(i, j) for i in range(n) for j in range(i+1, n) if current[i] & (1 << j)]
        random.shuffle(existing)
        for edge in existing[:2]:
            remove_edge(current, edge[0], edge[1])
        
        current, ls_count = local_search(current, deadline - 5)
        if ls_count > best_count:
            best_count = ls_count
            best_adj = list(current)

    # Final greedy add
    final_adj = list(best_adj)
    missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (final_adj[i] & (1 << j))]
    random.shuffle(missing)
    for u, v in missing:
        if time.time() > deadline - 1:
            break
        common = final_adj[u] & final_adj[v] & ~(1 << u) & ~(1 << v)
        if common:
            continue
        new_adj = list(final_adj)
        add_edge(new_adj, u, v)
        if not is_3_colorable(new_adj):
            add_edge(final_adj, u, v)

    final_count = edge_count(final_adj)
    if final_count > best_count:
        best_count = final_count
        best_adj = final_adj

    # Return result
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
