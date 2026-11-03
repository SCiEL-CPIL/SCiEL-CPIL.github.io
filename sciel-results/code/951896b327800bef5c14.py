<<<<<<< SEARCH
def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    
    # Precompute
    t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks = _precompute(v, k, t)
    
    # Multiple restarts with different seeds
    best_result = None
    best_count = float('inf')
    
    seed = 0
    max_restarts = 50
    while seed < max_restarts:
        if time.time() - start_time > deadline - 15.0:
            break
        
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, seed)
        
        # Remove redundant blocks
        selected = _remove_redundant(selected, block_t_indices, n_t, start_time, deadline)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_result = list(selected)
        
        # Check time for next restart
        if time.time() - start_time > deadline - 20.0:
            break
        
        seed += 1
    
    if best_result is None:
        # Fallback
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, 0)
        selected = _remove_redundant(selected, block_t_indices, n_t, start_time, deadline)
        best_result = selected
    
    return [list(k_subsets[bi]) for bi in best_result]
=======
def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Precompute
    t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks = _precompute(v, k, t)
    
    # Multiple restarts with different seeds
    best_result = None
    best_count = float('inf')
    
    seed = 0
    max_restarts = 80
    while seed < max_restarts:
        if time.time() - start_time > deadline - 20.0:
            break
        
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, seed)
        
        # Remove redundant blocks
        selected = _remove_redundant(selected, block_t_indices, n_t, start_time, deadline)
        
        # Local search refinement
        if time.time() - start_time < deadline - 25.0:
            selected = _local_search_refine(selected, block_t_indices, n_t, t_to_blocks, start_time, deadline)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_result = list(selected)
        
        # Check time for next restart
        if time.time() - start_time > deadline - 25.0:
            break
        
        seed += 1
    
    if best_result is None:
        # Fallback
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, 0)
        selected = _remove_redundant(selected, block_t_indices, n_t, start_time, deadline)
        best_result = selected
    
    return [list(k_subsets[bi]) for bi in best_result]
>>>>>>> REPLACE
