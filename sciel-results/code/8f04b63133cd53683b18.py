import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    # 1. Map edges to indices
    edge_map = {}
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[(u, v)] = idx
            edge_map[(v, u)] = idx
            edge_list.append((u, v))
            idx += 1
            
    # 2. Precompute Q3 patterns
    std_q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    local_patterns = set()
    axis_perms = list(itertools.permutations([0, 1, 2]))
    
    for perm in axis_perms:
        for flip_mask in range(8):
            def transform(v):
                b0 = (v >> 0) & 1
                b1 = (v >> 1) & 1
                b2 = (v >> 2) & 1
                
                b0 ^= (flip_mask >> 0) & 1
                b1 ^= (flip_mask >> 1) & 1
                b2 ^= (flip_mask >> 2) & 1
                
                new_v = 0
                if b0: new_v |= (1 << perm[0])
                if b1: new_v |= (1 << perm[1])
                if b2: new_v |= (1 << perm[2])
                
                return new_v
            
            mapped_vertices = [transform(i) for i in range(8)]
            if len(set(mapped_vertices)) != 8:
                continue
                
            pattern_edges = set()
            for u, v in std_q3_edges:
                mu = transform(u)
                mv = transform(v)
                if mu > mv:
                    mu, mv = mv, mu
                pattern_edges.add((mu, mv))
            
            local_patterns.add(frozenset(pattern_edges))
            
    local_patterns_list = list(local_patterns)
    
    # 3. Precompute global Q3 masks
    all_q3_masks = []
    subsets_8 = list(itertools.combinations(range(n), 8))
    
    for s in subsets_8:
        local_to_global = list(s)
        for pattern in local_patterns_list:
            mask = 0
            valid = True
            for u, v in pattern:
                gu = local_to_global[u]
                gv = local_to_global[v]
                if gu > gv:
                    gu, gv = gv, gu
                e_idx = edge_map.get((gu, gv))
                if e_idx is None:
                    valid = False
                    break
                mask |= (1 << e_idx)
            if valid:
                all_q3_masks.append(mask)
                
    num_q3s = len(all_q3_masks)
    
    # 4. Precompute edge_to_q3_indices
    edge_to_q3_indices = [[] for _ in range(total_edges)]
    for i, mask in enumerate(all_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            edge_to_q3_indices[e_idx].append(i)
            m ^= lsb
            
    # 5. Search
    current_mask = (1 << total_edges) - 1
    q3_counts = [12] * num_q3s
    active_q3s = set(range(num_q3s))
    
    best_mask = 0
    best_count = 0
    
    def update_best():
        nonlocal best_mask, best_count
        cnt = bin(current_mask).count('1')
        if cnt > best_count:
            best_count = cnt
            best_mask = current_mask
            
    max_iterations = 200000 # Increase iterations, operations are cheap
    for it in range(max_iterations):
        if time.time() > deadline - 2:
            break
            
        if not active_q3s:
            # Current graph is Q3-free
            update_best()
            
            # Try to add edges
            missing_edges = [e for e in range(total_edges) if not (current_mask & (1 << e))]
            if not missing_edges:
                break # Complete graph is valid?
                
            random.shuffle(missing_edges)
            added_any = False
            for e in missing_edges:
                if time.time() > deadline - 2:
                    break
                
                creates_q3 = False
                for q3_idx in edge_to_q3_indices[e]:
                    if q3_counts[q3_idx] == 11:
                        creates_q3 = True
                        break
                
                if not creates_q3:
                    # Add edge e
                    current_mask |= (1 << e)
                    for q3_idx in edge_to_q3_indices[e]:
                        q3_counts[q3_idx] += 1
                        if q3_counts[q3_idx] == 12:
                            active_q3s.add(q3_idx)
                    added_any = True
                    # If we created a Q3, revert
                    if active_q3s:
                        current_mask &= ~(1 << e)
                        for q3_idx in edge_to_q3_indices[e]:
                            q3_counts[q3_idx] -= 1
                            if q3_counts[q3_idx] == 11:
                                if q3_idx in active_q3s:
                                    active_q3s.remove(q3_idx)
                        break # Stop adding, go back to removal
                    # Else, continue trying other edges
            else:
                # If we finished the loop without breaking (i.e., no edge created a Q3)
                # We added some edges. 
                # If active_q3s is still empty, we can try to add more?
                # The loop structure above breaks if an edge creates a Q3.
                # If no edge creates a Q3, we added all possible edges?
                # No, we only iterated through a shuffled list of missing edges.
                # If we added one, we continue to the next in the list.
                # If we added one, and it didn't create a Q3, we continue.
                # This is fine.
                
            if not added_any:
                # No edges could be added. Local optimum.
                # Perturb: remove a random edge
                present_edges = [e for e in range(total_edges) if (current_mask & (1 << e))]
                if present_edges:
                    e_remove = random.choice(present_edges)
                    current_mask &= ~(1 << e_remove)
                    for q3_idx in edge_to_q3_indices[e_remove]:
                        q3_counts[q3_idx] -= 1
                        if q3_counts[q3_idx] == 11:
                            if q3_idx in active_q3s:
                                active_q3s.remove(q3_idx)
                else:
                    break
        else:
            # Remove edges to break Q3s
            # Pick a random active Q3
            q3_idx = next(iter(active_q3s))
            q3_mask = all_q3_masks[q3_idx]
            
            # Get edges in this Q3
            edges_in_q3 = []
            m = q3_mask
            while m:
                lsb = m & -m
                e_idx = lsb.bit_length() - 1
                edges_in_q3.append(e_idx)
                m ^= lsb
            
            # Pick edge with highest "impact" (in most active Q3s)
            best_edge = None
            best_score = -1
            for e in edges_in_q3:
                score = 0
                for q in edge_to_q3_indices[e]:
                    if q in active_q3s:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                best_edge = random.choice(edges_in_q3)
                
            # Remove best_edge
            current_mask &= ~(1 << best_edge)
            for q in edge_to_q3_indices[best_edge]:
                q3_counts[q] -= 1
                if q3_counts[q] == 11:
                    if q in active_q3s:
                        active_q3s.remove(q)
                        
    if not active_q3s:
        update_best()
        
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        e_idx = lsb.bit_length() - 1
        u, v = edge_list[e_idx]
        result.append([u, v])
        m ^= lsb
        
    return result
