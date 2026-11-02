<<<<<<< SEARCH
    best_blocks = None
    best_len = float('inf')
    
    # Main loop: Multiple restarts
    seed_counter = 0
    while time.time() < deadline - 10:
        # Generate initial covering
        blocks = greedy_cover(seed_counter)
        
        # Refine with local search
        # Use a different seed for local search to ensure variety
        refined_blocks = local_search(blocks, seed_counter + 1000)
        
        if len(refined_blocks) < best_len:
            best_len = len(refined_blocks)
            best_blocks = refined_blocks
        
        seed_counter += 1
        
        # If we have time, do more local search on the best solution found so far
        if best_blocks is not None and time.time() < deadline - 20:
            # Try to improve the best solution further
            current_best_copy = [list(b) for b in best_blocks]
            improved_best = local_search(current_best_copy, seed_counter + 2000)
            if len(improved_best) < best_len:
                best_len = len(improved_best)
                best_blocks = improved_best

    return best_blocks if best_blocks is not None else []
=======
    best_blocks = None
    best_len = float('inf')
    
    # Main loop: Multiple restarts
    seed_counter = 0
    while time.time() < deadline - 10:
        # Generate initial covering
        blocks = greedy_cover(seed_counter)
        
        # Refine with local search
        # Use a different seed for local search to ensure variety
        refined_blocks = local_search(blocks, seed_counter + 1000)
        
        if len(refined_blocks) < best_len:
            best_len = len(refined_blocks)
            best_blocks = refined_blocks
        
        seed_counter += 1
        
        # If we have time, do more local search on the best solution found so far
        if best_blocks is not None and time.time() < deadline - 20:
            # Try to improve the best solution further
            current_best_copy = [list(b) for b in best_blocks]
            improved_best = local_search(current_best_copy, seed_counter + 2000)
            if len(improved_best) < best_len:
                best_len = len(improved_best)
                best_blocks = improved_best

    return best_blocks if best_blocks is not None else []
>>>>>>> REPLACE
