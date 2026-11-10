"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random

def construct():
    n = 256
    
    # Precompute all 4-term APs
    four_aps = []
    for d in range(1, (n-1)//3 + 1):
        for a in range(1, n - 3*d + 1):
            four_aps.append((a, a+d, a+2*d, a+3*d))
    
    # For each vertex, find which 4-APs it belongs to
    vertex_to_4aps = {i: [] for i in range(1, n+1)}
    for idx, ap in enumerate(four_aps):
        for v in ap:
            vertex_to_4aps[v].append(idx)
            
    # Precompute 3-AP neighbors for incremental counting
    # neighbors3[v] = list of (u, w) such that {v, u, w} is a 3-AP
    neighbors3 = {i: [] for i in range(1, n+1)}
    for d in range(1, n//2 + 1):
        for a in range(1, n - 2*d + 1):
            b = a + d
            c = a + 2*d
            # a, b, c form a 3-AP
            neighbors3[a].append((b, c))
            neighbors3[b].append((a, c))
            neighbors3[c].append((a, b))
            
    # Weights for greedy ordering
    weights = {}
    for x in range(1, n+1):
        weights[x] = len(neighbors3[x])
        
    sorted_vertices = sorted(range(1, n+1), key=lambda v: -weights[v])
    
    def count_3aps_full(s):
        s_set = set(s)
        count = 0
        for a in s:
            for d in range(1, (n - a)//2 + 1):
                if a + d in s_set and a + 2*d in s_set:
                    count += 1
        return count

    def try_local_search(selected, current_count):
        improved = True
        while improved:
            improved = False
            # Try adding a vertex
            for v in range(1, n+1):
                if v in selected:
                    continue
                # Check 4-AP conflict
                can_add = True
                for ap_idx in vertex_to_4aps[v]:
                    ap = four_aps[ap_idx]
                    others = [u for u in ap if u != v]
                    if all(u in selected for u in others):
                        can_add = False
                        break
                if not can_add:
                    continue
                
                # Calculate delta in 3-AP count
                delta = 0
                for u, w in neighbors3[v]:
                    if u in selected and w in selected:
                        delta += 1
                        
                if current_count + delta > current_count: # Always true if delta > 0
                    # Actually, we just want to add if it improves.
                    # If delta == 0, adding doesn't help 3-AP count, but might allow future adds?
                    # Usually we only add if it increases score.
                    if delta > 0:
                        selected.add(v)
                        current_count += delta
                        improved = True
                        break # Restart loop to check new possibilities
            if improved:
                continue
                
            # Try swapping: remove u, add v
            # This is expensive O(N^2). Limit to small N or specific candidates.
            # For now, let's try simple removals that enable additions.
            # Or just stick to additions.
            
            # Try 2-swaps?
            # Given time, let's just do single adds and single swaps.
            
            # Try swapping one out for one in
            best_swap_gain = 0
            best_swap_pair = None
            for u in list(selected):
                # Remove u
                # Count loss from removing u
                loss = 0
                for x, y in neighbors3[u]:
                    if x in selected and y in selected:
                        # Check if x,y are not u
                        loss += 1
                # Now try adding each v
                for v in range(1, n+1):
                    if v in selected:
                        continue
                    # Check if v can be added to (selected - {u})
                    temp_selected = selected - {u}
                    can_add = True
                    for ap_idx in vertex_to_4aps[v]:
                        ap = four_aps[ap_idx]
                        others = [x for x in ap if x != v]
                        if all(x in temp_selected for x in others):
                            can_add = False
                            break
                    if not can_add:
                        continue
                    
                    # Gain from adding v to temp_selected
                    gain = 0
                    for x, y in neighbors3[v]:
                        if x in temp_selected and y in temp_selected:
                            gain += 1
                            
                    net_gain = gain - loss
                    if net_gain > best_swap_gain:
                        best_swap_gain = net_gain
                        best_swap_pair = (u, v)
                        
            if best_swap_pair and best_swap_gain > 0:
                u, v = best_swap_pair
                selected.remove(u)
                selected.add(v)
                current_count += best_swap_gain
                improved = True
                
        return selected, current_count

    best_set = set()
    best_score = 0
    
    # Multiple restarts
    for restart in range(10):
        # Randomize weights slightly or shuffle tie-breakers
        if restart == 0:
            order = sorted_vertices
        else:
            # Perturb weights
            perturbed_weights = {v: weights[v] + random.random() for v in range(1, n+1)}
            order = sorted(range(1, n+1), key=lambda v: -perturbed_weights[v])
            
        selected = set()
        for v in order:
            can_add = True
            for ap_idx in vertex_to_4aps[v]:
                ap = four_aps[ap_idx]
                others = [u for u in ap if u != v]
                if all(u in selected for u in others):
                    can_add = False
                    break
            if can_add:
                selected.add(v)
                
        # Count 3-APs
        current_count = count_3aps_full(selected)
        
        # Local Search
        selected, current_count = try_local_search(selected, current_count)
        
        if current_count > best_score:
            best_score = current_count
            best_set = set(selected)
            
    return sorted(best_set)

# EVOLVE-BLOCK-END
