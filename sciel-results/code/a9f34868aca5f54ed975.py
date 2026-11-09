<<<<<<< SEARCH
def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m + 1
    best_result = None
    
    candidates = _generate_structured_values(n, max_val)
    
    random.seed(12345)
    
    for vals in candidates:
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        remaining = _DEADLINE - time.time()
        if remaining < 3:
            break
        
        # Step 1: Greedy edge selection
        edges, _, _ = _greedy_edge_selection(vals, n, m)
        
        # Step 2: Local search on edges
        edge_score, improved_edges = _local_search_edges(vals, edges, n, m, max_iter=3000)
        
        # Step 3: Polish values
        final_score, polished_values = _polish_values(vals, improved_edges, n, m, max_iter=1500)
        
        # Recompute with final values and edges
        final_score = _score_fast(polished_values, improved_edges)
        
        if final_score < best_score:
            best_score = final_score
            best_result = (polished_values, improved_edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
=======
def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m + 1
    best_result = None
    
    candidates = _generate_structured_values(n, max_val)
    
    # Add collision-density-based values
    collision_vals = _greedy_collision_value_selection(n, max_val)
    if collision_vals is not None:
        candidates.append(collision_vals)
    
    random.seed(12345)
    
    # Allocate time budget: use more time for promising candidates
    total_time_budget = _DEADLINE - time.time()
    num_candidates = len(candidates)
    time_per_candidate = total_time_budget / (num_candidates + 1)  # +1 for final polish
    
    for ci, vals in enumerate(candidates):
        if time.time() > _DEADLINE:
            break
        
        # Validate
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        remaining = _DEADLINE - time.time()
        if remaining < 3:
            break
        
        # Allocate more time to earlier candidates and collision-based ones
        is_collision_based = (ci == num_candidates - 1)
        if is_collision_based:
            edge_iter = 4000
            polish_iter = 2500
        else:
            edge_iter = 2500
            polish_iter = 1200
        
        # Step 1: Greedy edge selection
        edges, _, _ = _greedy_edge_selection(vals, n, m)
        
        # Step 2: Local search on edges
        edge_score, improved_edges = _local_search_edges(vals, edges, n, m, max_iter=edge_iter)
        
        # Step 3: Polish values
        final_score, polished_values = _polish_values(vals, improved_edges, n, m, max_iter=polish_iter)
        
        # Step 4: Re-optimize edges with polished values
        final_score2, final_edges = _local_search_edges(polished_values, improved_edges, n, m, max_iter=1500)
        
        # Step 5: Final polish with final edges
        final_score3, final_values = _polish_values(polished_values, final_edges, n, m, max_iter=800)
        
        # Recompute with final values and edges
        final_score = _score_fast(final_values, final_edges)
        
        if final_score < best_score:
            best_score = final_score
            best_result = (final_values, final_edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    # Final polish if we have time
    if time.time() < _DEADLINE - 5:
        final_score, polished_values = _polish_values(values, edges, n, m, max_iter=1000)
        if final_score < best_score:
            values = polished_values
            best_score = final_score
        # Re-optimize edges one more time
        if time.time() < _DEADLINE - 3:
            edge_score, final_edges = _local_search_edges(values, edges, n, m, max_iter=1000)
            if edge_score < best_score:
                edges = final_edges
                best_score = edge_score
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
>>>>>>> REPLACE
