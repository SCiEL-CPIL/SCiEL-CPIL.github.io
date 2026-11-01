<<<<<<< SEARCH
    # Main Loop
    num_restarts = 200
    seeds = [i for i in range(num_restarts)]
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        blocks, valid = greedy_construct(seed)
        if not valid:
            continue
            
        pruned = prune_redundant(blocks)
        
        if len(pruned) < best_count:
            # Run local search if we have time
            remaining_time = deadline - time.time()
            if remaining_time > 5.0:
                ls_blocks = local_search_swap(pruned, max_time=min(3.0, remaining_time * 0.5))
                if len(ls_blocks) < len(pruned):
                    pruned = ls_blocks
            
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_blocks = [list(k_subsets[i]) for i in pruned]
                # If we reach a very good solution, we can be more aggressive or stop early if time is tight
                # But keep searching for better.
=======
    # Main Loop
    num_restarts = 350
    seeds = list(range(num_restarts))
    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        blocks, valid = greedy_construct(seed)
        if not valid:
            continue
            
        pruned = prune_redundant(blocks)
        
        if len(pruned) < best_count:
            # Run local search if we have time
            remaining_time = deadline - time.time()
            if remaining_time > 5.0:
                ls_blocks = local_search_swap(pruned, max_time=min(4.0, remaining_time * 0.5))
                if len(ls_blocks) < len(pruned):
                    pruned = ls_blocks
            
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_blocks = [list(k_subsets[i]) for i in pruned]
>>>>>>> REPLACE
