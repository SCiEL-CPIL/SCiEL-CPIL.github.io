<<<<<<< SEARCH
    # Run multiple restarts
    best_overall_score = float('inf')
    best_overall_edges = None
    
    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        restart_deadline = time.time() + time_per_restart - 0.5
        
        # Initialize
        adj, deg, current_edges = build_initial_graph(42 + restart * 1000)
        tri_sum = compute_tri_sums(adj, deg)
        current_score = max(tri_sum)
        
        best_score = current_score
        best_adj = list(adj)
        best_deg = list(deg)
        best_edges = set(current_edges)
        
        rng = random.Random(42 + restart * 7919)
        
        T_start = 8.0
        T_end = 0.005
        
        # Precompute non-edges for faster sampling
        non_edges = get_non_edges(adj)
        
        i = 0
        n_iter_total = 400000
        
        while time.time() < restart_deadline:
            i += 1
            # Faster cooling in first half, slower in second half for better exploration
            progress = min(i, n_iter_total) / n_iter_total
            if progress < 0.5:
                T = T_start * (T_end / T_start) ** (progress * 2)
            else:
                T = T_start * (T_end / T_start) ** (0.5 + (progress - 0.5) * 1.5)
            
            # Move Selection
            # 1. Pick edge to remove - bias toward edges in high-sum triangles
            if current_score > 0:
                # Find triangles with max or near-max sum
                max_val = current_score
                # Consider triangles with sum >= max_val - 2 for more diversity
                threshold = max_val - 2
                candidates = [idx for idx, s in enumerate(tri_sum) if s >= threshold]
                
                if candidates and rng.random() < 0.7:
                    idx = rng.choice(candidates)
                    u, v, w = all_tris[idx]
                    # Try to find best edge to remove from this triangle
                    best_remove = find_best_edge_removal_for_triangle(u, v, w, adj, deg, current_edges)
                    if best_remove:
                        e_remove = best_remove
                    else:
                        e_remove = rng.choice([(u, v), (u, w), (v, w)])
                        e_remove = (min(e_remove), max(e_remove))
                else:
                    e_remove = rng.choice(list(current_edges))
                    e_remove = (min(e_remove), max(e_remove))
            else:
                e_remove = rng.choice(list(current_edges))
                e_remove = (min(e_remove), max(e_remove))
                
            # 2. Pick edge to add from non_edges list
            if not non_edges:
                break
            
            # Bias toward adding edges that connect lower-degree vertices
            if rng.random() < 0.4:
                # Weight by inverse degree product
                weights = []
                for ne in non_edges:
                    a, b = ne
                    w_val = 1.0 / (max(1, deg[a]) * max(1, deg[b]))
                    weights.append(w_val)
                total_w = sum(weights)
                if total_w > 0:
                    r = rng.random() * total_w
                    cumsum = 0
                    for ne, w_val in zip(non_edges, weights):
                        cumsum += w_val
                        if r <= cumsum:
                            e_add = ne
                            break
                    else:
                        e_add = rng.choice(non_edges)
                else:
                    e_add = rng.choice(non_edges)
            else:
                e_add = rng.choice(non_edges)
            
            if e_add == e_remove:
                continue
                
            u1, v1 = e_remove
            u2, v2 = e_add
            
            prev_score = current_score
            
            # Apply Move
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            deg[u1] -= 1
            deg[v1] -= 1
            current_edges.remove(e_remove)
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            deg[u2] += 1
            deg[v2] += 1
            current_edges.add(e_add)
            
            # Update non_edges list
            non_edges.remove(e_add)
            non_edges.append(e_remove)
            
            # Update Triangle Sums Incrementally
            # Triangles affected: those containing (u1,v1) or (u2,v2) or any of u1,v1,u2,v2
            update_indices = set()
            
            # Triangles containing the removed edge
            if e_remove in edge_to_tris:
                update_indices.update(edge_to_tris[e_remove])
            # Triangles containing the added edge
            if e_add in edge_to_tris:
                update_indices.update(edge_to_tris[e_add])
            # Triangles containing any affected vertex (for degree sum changes)
            for vert in (u1, v1, u2, v2):
                update_indices.update(vert_to_tris[vert])
            
            # Update sums for these indices
            for idx in update_indices:
                u, v, w = all_tris[idx]
                if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                    tri_sum[idx] = deg[u] + deg[v] + deg[w]
                else:
                    tri_sum[idx] = 0
                    
            current_score = max(tri_sum)
            
            # Acceptance
            delta = current_score - prev_score
            if delta <= 0 or (T > 0.01 and rng.random() < math.exp(-delta / max(T, 1e-10))):
                if current_score < best_score:
                    best_score = current_score
                    best_adj = list(adj)
                    best_deg = list(deg)
                    best_edges = set(current_edges)
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                deg[u1] += 1
                deg[v1] += 1
                current_edges.add(e_remove)
                
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                deg[u2] -= 1
                deg[v2] -= 1
                current_edges.remove(e_add)
                
                # Revert non_edges
                non_edges.remove(e_remove)
                non_edges.append(e_add)
                
                # Revert triangle sums
                for idx in update_indices:
                    u, v, w = all_tris[idx]
                    if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                        tri_sum[idx] = deg[u] + deg[v] + deg[w]
                    else:
                        tri_sum[idx] = 0
                        
                current_score = prev_score
                
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_edges = best_edges
=======
    # Run multiple restarts
    best_overall_score = float('inf')
    best_overall_edges = None
    
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        restart_deadline = time.time() + time_per_restart - 0.5
        
        # Initialize with different seeds for diversity
        adj, deg, current_edges = build_initial_graph(42 + restart * 1000)
        tri_sum = compute_tri_sums_fast(adj, deg)
        current_score = max(tri_sum)
        
        best_score = current_score
        best_adj = list(adj)
        best_deg = list(deg)
        best_edges = set(current_edges)
        
        rng = random.Random(42 + restart * 7919)
        
        T_start = 6.0
        T_end = 0.001
        
        # Precompute non-edges for faster sampling
        non_edges = get_non_edges(adj)
        
        i = 0
        n_iter_total = 500000
        
        # Track candidate triangles for faster selection
        # Use a simple approach: maintain sorted list of top triangle sums
        
        while time.time() < restart_deadline:
            i += 1
            # Geometric cooling
            progress = min(i, n_iter_total) / n_iter_total
            T = T_start * (T_end / T_start) ** progress
            
            # Move Selection
            # 1. Pick edge to remove - bias toward edges in high-sum triangles
            if current_score > 0:
                # Find triangles with max or near-max sum
                max_val = current_score
                # Consider triangles with sum >= max_val - 2 for more diversity
                threshold = max_val - 2
                # Efficiently find candidates
                candidates = []
                for idx, s in enumerate(tri_sum):
                    if s >= threshold:
                        candidates.append(idx)
                
                if candidates and rng.random() < 0.6:
                    idx = rng.choice(candidates)
                    u, v, w = all_tris[idx]
                    # Try to find best edge to remove from this triangle
                    best_remove = find_best_edge_removal_for_triangle(u, v, w, adj, deg, current_edges)
                    if best_remove:
                        e_remove = best_remove
                    else:
                        e_remove = rng.choice([(u, v), (u, w), (v, w)])
                        e_remove = (min(e_remove), max(e_remove))
                else:
                    # Random edge from current edges
                    e_remove = rng.choice(list(current_edges))
                    e_remove = (min(e_remove), max(e_remove))
            else:
                e_remove = rng.choice(list(current_edges))
                e_remove = (min(e_remove), max(e_remove))
                
            # 2. Pick edge to add from non_edges list
            if not non_edges:
                break
            
            # Bias toward adding edges that connect lower-degree vertices
            # Use a faster weighted selection
            if rng.random() < 0.3:
                # Weight by inverse degree product - use reservoir sampling approach
                # For small lists, just compute weights
                best_w = 0
                total_w = 0.0
                for ne in non_edges:
                    a, b = ne
                    w_val = 1.0 / (max(1, deg[a]) * max(1, deg[b]))
                    total_w += w_val
                if total_w > 0:
                    r = rng.random() * total_w
                    cumsum = 0.0
                    for ne in non_edges:
                        a, b = ne
                        w_val = 1.0 / (max(1, deg[a]) * max(1, deg[b]))
                        cumsum += w_val
                        if r <= cumsum:
                            e_add = ne
                            break
                    else:
                        e_add = rng.choice(non_edges)
                else:
                    e_add = rng.choice(non_edges)
            else:
                e_add = rng.choice(non_edges)
            
            if e_add == e_remove:
                continue
                
            u1, v1 = e_remove
            u2, v2 = e_add
            
            prev_score = current_score
            
            # Apply Move
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            deg[u1] -= 1
            deg[v1] -= 1
            current_edges.remove(e_remove)
            
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            deg[u2] += 1
            deg[v2] += 1
            current_edges.add(e_add)
            
            # Update non_edges list - use O(1) swap-remove
            # Find index of e_add in non_edges
            try:
                idx_remove = non_edges.index(e_add)
                non_edges[idx_remove] = e_remove
                # Move e_remove to end for O(1) removal later if needed
            except ValueError:
                non_edges.append(e_remove)
            
            # Update Triangle Sums Incrementally
            # Triangles affected: those containing (u1,v1) or (u2,v2) or any of u1,v1,u2,v2
            update_indices = set()
            
            # Triangles containing the removed edge
            if e_remove in edge_to_tris:
                update_indices.update(edge_to_tris[e_remove])
            # Triangles containing the added edge
            if e_add in edge_to_tris:
                update_indices.update(edge_to_tris[e_add])
            # Triangles containing any affected vertex (for degree sum changes)
            for vert in (u1, v1, u2, v2):
                update_indices.update(vert_to_tris[vert])
            
            # Update sums for these indices
            new_max = 0
            for idx in update_indices:
                u, v, w = all_tris[idx]
                if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                    s = deg[u] + deg[v] + deg[w]
                    tri_sum[idx] = s
                    if s > new_max:
                        new_max = s
                else:
                    tri_sum[idx] = 0
            
            # Also check if any other triangles have higher sum
            # Only need to check if new_max < prev_score
            if new_max < prev_score:
                # Need to check all triangles
                for idx in range(num_tris):
                    if tri_sum[idx] > new_max:
                        new_max = tri_sum[idx]
            
            current_score = new_max
            
            # Acceptance
            delta = current_score - prev_score
            if delta <= 0 or (T > 0.01 and rng.random() < math.exp(-delta / max(T, 1e-10))):
                if current_score < best_score:
                    best_score = current_score
                    best_adj = list(adj)
                    best_deg = list(deg)
                    best_edges = set(current_edges)
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                deg[u1] += 1
                deg[v1] += 1
                current_edges.add(e_remove)
                
                adj[u2] &= ~(1 << v2)
                adj[v2] &= ~(1 << u2)
                deg[u2] -= 1
                deg[v2] -= 1
                current_edges.remove(e_add)
                
                # Revert non_edges
                try:
                    idx_revert = non_edges.index(e_remove)
                    non_edges[idx_revert] = e_add
                except ValueError:
                    non_edges.remove(e_remove)
                    non_edges.append(e_add)
                
                # Revert triangle sums
                for idx in update_indices:
                    u, v, w = all_tris[idx]
                    if (adj[u] >> v) & 1 and (adj[u] >> w) & 1 and (adj[v] >> w) & 1:
                        tri_sum[idx] = deg[u] + deg[v] + deg[w]
                    else:
                        tri_sum[idx] = 0
                        
                current_score = prev_score
                
        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_edges = best_edges
>>>>>>> REPLACE

