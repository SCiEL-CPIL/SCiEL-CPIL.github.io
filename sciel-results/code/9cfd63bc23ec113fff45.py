"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    from random import Random
    start_time = time.time()
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)
    
    def color_graph(edges, deadline):
        if not edges:
            return []
        
        m = len(edges)
        
        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Precompute covers: for edge (u,v), cover = {u,v} ∪ N(u) ∪ N(v)
        # Two edges conflict if one has an endpoint in the other's cover
        covers = []
        for u, v in edges:
            cov = {u, v}
            cov.update(adj.get(u, set()))
            cov.update(adj.get(v, set()))
            covers.append(cov)
        
        # Build conflict bitmasks
        conflict_masks = [0] * m
        for i in range(m):
            cov_i = covers[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov_i or v_j in cov_i:
                    mask |= (1 << j)
            conflict_masks[i] = mask
        
        # Conflict degree
        conflict_deg = [bin(cm).count('1') for cm in conflict_masks]
        
        # --- Strategy 1: Greedy coloring with multiple orderings ---
        def greedy_color(order):
            color = [-1] * m
            color_used_by = []  # bitmask of edges per color
            
            max_color = 0
            for i in order:
                cm = conflict_masks[i]
                taken = set()
                for c, mask in enumerate(color_used_by):
                    if cm & mask:
                        taken.add(c)
                
                c = 0
                while c in taken:
                    c += 1
                
                color[i] = c
                if c >= len(color_used_by):
                    color_used_by.append(0)
                color_used_by[c] |= (1 << i)
                if c > max_color:
                    max_color = c
            
            return color, max_color
        
        # --- Strategy 2: Repeated maximal independent set extraction ---
        def mis_color(remaining_mask, rng):
            color = [-1] * m
            color_count = 0
            uncolored = remaining_mask
            
            while uncolored and time.time() < deadline:
                # Find a large independent set among uncolored edges
                # Greedy: pick edges in order of fewest conflicts with remaining uncolored
                uncolored_indices = [i for i in range(m) if (uncolored >> i) & 1]
                
                # Sort by conflict degree within uncolored set
                def conf_in_uncolored(i):
                    cm = conflict_masks[i] & uncolored
                    return bin(cm).count('1')
                
                uncolored_indices.sort(key=conf_in_uncolored)
                
                # Greedily build independent set
                independent_set = []
                available = uncolored
                for i in uncolored_indices:
                    if not (available >> i) & 1:
                        continue
                    # Check if i conflicts with any already in independent_set
                    can_add = True
                    for j in independent_set:
                        if (conflict_masks[i] >> j) & 1:
                            can_add = False
                            break
                    if can_add:
                        independent_set.append(i)
                        # Remove i and all its conflicts from available
                        available &= ~(conflict_masks[i] | (1 << i))
                
                if not independent_set:
                    # Shouldn't happen, but safety break
                    break
                
                # Assign this color class
                for i in independent_set:
                    color[i] = color_count
                
                # Remove from uncolored
                for i in independent_set:
                    uncolored &= ~(1 << i)
                
                color_count += 1
            
            # If any edges remain uncolored (shouldn't happen), color them greedily
            remaining = [i for i in range(m) if color[i] == -1]
            for i in remaining:
                taken = set()
                cm = conflict_masks[i]
                for j in range(m):
                    if color[j] != -1 and (cm >> j) & 1:
                        taken.add(color[j])
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                color_count = max(color_count, c + 1)
            
            return color, color_count
        
        # Generate orderings
        orderings = []
        
        # 1. Descending conflict degree
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i], reverse=True))
        
        # 2. Ascending conflict degree
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i]))
        
        # 3. Degree-based ordering
        orderings.append(sorted(range(m), key=lambda i: len(adj.get(edges[i][0], set())) + len(adj.get(edges[i][1], set())), reverse=True))
        
        # 4. Random orderings
        rng = Random(42)
        for _ in range(30):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)
        
        # 5. Mixed orderings: shuffle among high-conflict first
        for _ in range(20):
            high = [i for i in range(m) if conflict_deg[i] > m // 2]
            low = [i for i in range(m) if conflict_deg[i] <= m // 2]
            rng.shuffle(high)
            rng.shuffle(low)
            orderings.append(high + low)
        
        best_color = None
        best_max_color = m + 1
        
        # Try greedy colorings
        for order in orderings:
            if time.time() > deadline:
                break
            color, max_color = greedy_color(order)
            if max_color < best_max_color:
                best_max_color = max_color
                best_color = color[:]
        
        # Try MIS-based coloring with several random seeds
        full_mask = (1 << m) - 1
        for seed in range(20):
            if time.time() > deadline:
                break
            rng_mis = Random(seed)
            color, max_color = mis_color(full_mask, rng_mis)
            if max_color < best_max_color:
                best_max_color = max_color
                best_color = color[:]
        
        # Local search: try to reduce colors by recoloring
        if best_color is not None and time.time() < deadline:
            # Try to merge color classes where possible
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                improved = False
                # For each pair of colors, check if we can merge them
                colors_used = sorted(set(best_color))
                for c1_idx, c1 in enumerate(colors_used[:-1]):
                    for c2 in colors_used[c1_idx+1:]:
                        # Check if edges with c1 and c2 can be merged
                        # i.e., no two edges (one c1, one c2) conflict
                        c1_edges = [i for i in range(m) if best_color[i] == c1]
                        c2_edges = [i for i in range(m) if best_color[i] == c2]
                        
                        can_merge = True
                        for i in c1_edges:
                            for j in c2_edges:
                                if (conflict_masks[i] >> j) & 1:
                                    can_merge = False
                                    break
                            if not can_merge:
                                break
                        
                        if can_merge:
                            # Merge: recolor c2 edges to c1
                            for i in c2_edges:
                                best_color[i] = c1
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if not improved:
                    break
        
        return best_color
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        elapsed = time.time() - start_time
        remaining = 145 - elapsed
        if remaining < 5:
            # Fallback to simple greedy
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue
        
        # Allocate time proportional to graph size
        n, d, seed = spec
        time_budget = min(remaining / 3, 145 - elapsed)
        deadline = start_time + 145 - 5  # 5s safety margin
        
        edges = instance(*spec)
        colors = color_graph(edges, deadline)
        out.append(colors)
        
    return out
# EVOLVE-BLOCK-END
