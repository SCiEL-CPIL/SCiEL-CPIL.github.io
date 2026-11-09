<<<<<<< SEARCH
    # Optimization Loop
    # We try to add edges. If we can't, we try swaps.
    # To avoid O(N^2) swap checks being too slow, we limit iterations or use randomization.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Add edges
        for e in range(num_edges_total):
            if removed_edges[e]:
                if can_add(e):
                    removed_edges[e] = False
                    improved = True
        
        if time.time() > deadline:
            break
            
        # Phase 2: Swaps (1-for-1)
        # If we couldn't add any edge, try to swap.
        # This phase is more expensive. We might want to limit it or run it less frequently.
        # Let's try to find a swap that allows a future addition or just maintains density while changing structure.
        # Actually, if we are in a local optimum for "add", a swap might open up new "add" possibilities.
        
        # For each removed edge v, if it cannot be added, find why.
        # The "blockers" are Q3s where all other edges are present.
        # If we remove one of those other edges (u), v might become addable.
        
        # To keep it fast, we'll iterate through removed edges and try a few swaps.
        # We'll use a simple heuristic: try swapping v with u if u is in a blocking Q3 for v.
        
        # Precompute blocking Q3s for each removed edge? That's dynamic.
        # Let's just try: for each v (removed), if not can_add(v), pick a random blocking Q3, pick a random u from it, try swap.
        
        # To be deterministic and avoid randomness overhead, we'll do a single pass of potential swaps.
        # We'll collect candidates.
        
        # Note: A swap removes u and adds v. Net change 0.
        # It's only useful if it leads to more adds later.
        # Given time constraints, we'll perform a limited number of swap attempts.
        
        swap_attempts = 0
        max_swap_attempts = 50 # Limit per pass to avoid slowdown
        
        for v in range(num_edges_total):
            if not removed_edges[v]:
                continue
            if can_add(v):
                # Should have been added in Phase 1, but maybe state changed?
                # In our loop, we don't update 'improved' inside the swap logic immediately to allow multiple swaps?
                # No, let's stick to Phase 1 for adds.
                pass
            
            # Check if v is blocked
            blocked = False
            blocking_q3 = -1
            for q3_idx in q3s_containing_edge[v]:
                is_complete = True
                for other_e in q3_edge_lists[q3_idx]:
                    if other_e == v:
                        continue
                    if removed_edges[other_e]:
                        is_complete = False
                        break
                if is_complete:
                    blocked = True
                    blocking_q3 = q3_idx
                    break
            
            if blocked and swap_attempts < max_swap_attempts:
                # Try to remove one edge u from blocking_q3 to unblock v
                # We want to remove an edge u that is "cheap" to remove or whose removal doesn't break other things?
                # Since we are maintaining validity, removing u is always safe (makes graph sparser).
                # The risk is that we lose an edge. But if we add v, we are neutral.
                # The benefit is structural change.
                
                # Pick the first edge u in blocking_q3 (excluding v) that is present (not removed)
                # Actually, all edges in blocking_q3 except v are present by definition of "blocking".
                # So we can pick any u in blocking_q3 \ {v}.
                
                # To be strategic, maybe pick u that is in the most Q3s? No, that would be bad to remove.
                # Pick u that is in the FEWEST Q3s? That might be safer.
                # Or just pick the first one.
                
                candidate_u = -1
                for u in q3_edge_lists[blocking_q3]:
                    if u != v:
                        candidate_u = u
                        break
                
                if candidate_u != -1:
                    # Perform swap: remove candidate_u, add v
                    # Check if adding v is safe *after* removing candidate_u.
                    # Since candidate_u is in the blocking Q3, removing it breaks that Q3.
                    # Are there other Q3s containing v that are still complete?
                    # If yes, the swap is invalid (we'd create a different Q3).
                    
                    safe_swap = True
                    for q3_idx_check in q3s_containing_edge[v]:
                        # Check if this Q3 is complete after removing candidate_u
                        # It is complete if all edges in Q3 except v are present.
                        # If candidate_u is in this Q3, it is now removed, so Q3 is broken.
                        # If candidate_u is NOT in this Q3, we need to check if all other edges are present.
                        
                        is_complete = True
                        for other_e in q3_edge_lists[q3_idx_check]:
                            if other_e == v:
                                continue
                            if other_e == candidate_u:
                                # This edge is being removed, so Q3 is not complete
                                is_complete = False
                                break
                            if removed_edges[other_e]:
                                is_complete = False
                                break
                        if is_complete:
                            safe_swap = False
                            break
                    
                    if safe_swap:
                        removed_edges[candidate_u] = True
                        removed_edges[v] = False
                        improved = True
                        swap_attempts += 1
                        # Restart the add phase? Or just continue.
                        # Continuing might allow more swaps.
                        # But we should probably break to re-evaluate adds.
                        # Actually, let's just break the inner loop and let the outer while handle it.
                        # We set improved=True, so it will loop again.
                        # Breaking the for v loop ensures we don't do multiple swaps in one "pass" which might be complex.
                        # Let's break the for loop.
                        break

    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges
=======
    # Optimization Loop
    # We try to add edges. If we can't, we try swaps.
    # To avoid O(N^2) swap checks being too slow, we limit iterations or use randomization.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Add edges
        for e in range(num_edges_total):
            if removed_edges[e]:
                if can_add(e):
                    removed_edges[e] = False
                    improved = True
        
        if time.time() > deadline:
            break
            
        # Phase 2: 1-for-1 Swaps
        swap_attempts = 0
        max_swap_attempts = 50 # Limit per pass to avoid slowdown
        
        for v in range(num_edges_total):
            if time.time() > deadline:
                break
            if not removed_edges[v]:
                continue
            
            # Check if v is blocked
            blocked = False
            blocking_q3 = -1
            for q3_idx in q3s_containing_edge[v]:
                is_complete = True
                for other_e in q3_edge_lists[q3_idx]:
                    if other_e == v:
                        continue
                    if removed_edges[other_e]:
                        is_complete = False
                        break
                if is_complete:
                    blocked = True
                    blocking_q3 = q3_idx
                    break
            
            if blocked and swap_attempts < max_swap_attempts:
                # Try to remove one edge u from blocking_q3 to unblock v
                # Pick u that is in the FEWEST Q3s to minimize disruption
                best_u = -1
                min_q3_count = float('inf')
                for u in q3_edge_lists[blocking_q3]:
                    if u != v and not removed_edges[u]:
                        q3_count = len(q3s_containing_edge[u])
                        if q3_count < min_q3_count:
                            min_q3_count = q3_count
                            best_u = u
                
                if best_u != -1:
                    # Perform swap: remove best_u, add v
                    safe_swap = True
                    for q3_idx_check in q3s_containing_edge[v]:
                        is_complete = True
                        for other_e in q3_edge_lists[q3_idx_check]:
                            if other_e == v:
                                continue
                            if other_e == best_u:
                                is_complete = False
                                break
                            if removed_edges[other_e]:
                                is_complete = False
                                break
                        if is_complete:
                            safe_swap = False
                            break
                    
                    if safe_swap:
                        removed_edges[best_u] = True
                        removed_edges[v] = False
                        improved = True
                        swap_attempts += 1
                        break  # Restart add phase
        
        if time.time() > deadline:
            break
        
        # Phase 3: 2-for-2 Swaps (to escape 1-swap local optima)
        if not improved:
            two_swap_found = False
            # Try to find two removed edges v1, v2 and two present edges u1, u2
            # such that removing u1, u2 and adding v1, v2 is valid
            removed_list = [e for e in range(num_edges_total) if removed_edges[e]]
            present_list = [e for e in range(num_edges_total) if not removed_edges[e]]
            
            # Limit the search to avoid combinatorial explosion
            max_v_pairs = 20
            v_pairs_checked = 0
            
            for i in range(min(len(removed_list), 20)):
                if time.time() > deadline:
                    break
                v1 = removed_list[i]
                for j in range(i + 1, min(len(removed_list), 20)):
                    if time.time() > deadline:
                        break
                    v2 = removed_list[j]
                    v_pairs_checked += 1
                    
                    # Check if {v1, v2} can be added together (given current state)
                    # First, check if they individually have blockers
                    # Then try to find u1, u2 that unblock both
                    
                    # For simplicity, try a few present edges as u1, u2
                    # Pick u1 from a blocking Q3 of v1, u2 from a blocking Q3 of v2
                    blocking_q3_v1 = -1
                    for q3_idx in q3s_containing_edge[v1]:
                        is_complete = True
                        for other_e in q3_edge_lists[q3_idx]:
                            if other_e == v1:
                                continue
                            if removed_edges[other_e]:
                                is_complete = False
                                break
                        if is_complete:
                            blocking_q3_v1 = q3_idx
                            break
                    
                    blocking_q3_v2 = -1
                    for q3_idx in q3s_containing_edge[v2]:
                        is_complete = True
                        for other_e in q3_edge_lists[q3_idx]:
                            if other_e == v2:
                                continue
                            if removed_edges[other_e]:
                                is_complete = False
                                break
                        if is_complete:
                            blocking_q3_v2 = q3_idx
                            break
                    
                    if blocking_q3_v1 == -1 or blocking_q3_v2 == -1:
                        continue
                    
                    # Try a few u1 candidates from blocking_q3_v1
                    u1_candidates = [u for u in q3_edge_lists[blocking_q3_v1] if u != v1 and not removed_edges[u]][:3]
                    u2_candidates = [u for u in q3_edge_lists[blocking_q3_v2] if u != v2 and not removed_edges[u]][:3]
                    
                    for u1 in u1_candidates:
                        if time.time() > deadline:
                            break
                        for u2 in u2_candidates:
                            if u1 == u2:
                                continue
                            # Check if removing u1, u2 and adding v1, v2 is valid
                            # Temporarily mark u1, u2 as removed and v1, v2 as present
                            removed_edges[u1] = True
                            removed_edges[u2] = True
                            removed_edges[v1] = False
                            removed_edges[v2] = False
                            
                            # Check validity
                            valid = True
                            for e in [v1, v2]:
                                for q3_idx in q3s_containing_edge[e]:
                                    is_complete = True
                                    for other_e in q3_edge_lists[q3_idx]:
                                        if other_e == v1 or other_e == v2:
                                            continue
                                        if other_e == u1 or other_e == u2:
                                            is_complete = False
                                            break
                                        if removed_edges[other_e]:
                                            is_complete = False
                                            break
                                    if is_complete:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            
                            # Revert
                            removed_edges[u1] = False
                            removed_edges[u2] = False
                            removed_edges[v1] = True
                            removed_edges[v2] = True
                            
                            if valid:
                                # Perform the 2-swap
                                removed_edges[u1] = True
                                removed_edges[u2] = True
                                removed_edges[v1] = False
                                removed_edges[v2] = False
                                improved = True
                                two_swap_found = True
                                break
                        if two_swap_found:
                            break
                    if two_swap_found:
                        break
                    if v_pairs_checked >= max_v_pairs:
                        break

    # Construct result
    result_edges = []
    for e in range(num_edges_total):
        if not removed_edges[e]:
            u, v = inv_edge_map[e]
            result_edges.append([u, v])
            
    return result_edges
>>>>>>> REPLACE
