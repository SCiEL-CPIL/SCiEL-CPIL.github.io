<<<<<<< SEARCH
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 105  # 15s buffer for safety
    
    seed_counter = 0
    while time.time() < deadline:
        blocks = greedy_cover(seed_counter)
        blocks = trim_blocks(blocks)
        blocks = local_search_trim(blocks, seed_counter, time_limit=1.0)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
=======
    # Main loop
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 105  # 15s buffer for safety
    
    seed_counter = 0
    while time.time() < deadline:
        # Alternate between greedy and ABF-KP for diversity
        if seed_counter % 2 == 0:
            blocks = greedy_cover(seed_counter)
        else:
            blocks = abfkp_cover(seed_counter, time_budget=4.0)
        
        blocks = trim_blocks(blocks)
        blocks = local_search_trim(blocks, seed_counter, time_limit=1.5)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
            
        if best_len == 69:
            break
            
        seed_counter += 1
    
    # Final polish: try more aggressive local search on the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        polished = local_search_trim(best_blocks, 99999, time_limit=10.0)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)
    
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        return []
>>>>>>> REPLACE
