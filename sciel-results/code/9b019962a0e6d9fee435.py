<<<<<<< SEARCH
    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 12
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Use bitmask greedy for speed
        selected = _greedy_cover_bitmask(block_covers_bitmasks, n_t, sub_to_blocks, deadline, 
                                          randomize=(restart > 0), top_k=10)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search (includes 2-exchange)
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
=======
    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 15
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Use bitmask greedy for speed
        selected = _greedy_cover_bitmask(block_covers_bitmasks, n_t, sub_to_blocks, deadline, 
                                          randomize=(restart > 0), top_k=12)
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_covers, n_t, deadline)
        
        # Local search (includes 2-exchange)
        selected = _local_search(selected, block_covers, n_t, sub_to_blocks, deadline)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = set(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            best_selected = _local_search(best_selected, block_covers, n_t, sub_to_blocks, deadline)
            if len(best_selected) < best_size:
                best_size = len(best_selected)
>>>>>>> REPLACE

