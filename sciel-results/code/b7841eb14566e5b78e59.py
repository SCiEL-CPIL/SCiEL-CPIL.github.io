<<<<<<< SEARCH
    # Phase 2: Critical word swap on the best solution
    if best_codewords is not None:
        for seed in [42, 1, 7, 13]:
            if time.time() >= deadline - 5.0:
                break
            cw2 = _critical_word_swap(best_codewords, inc, neighbor_map, total_space, deadline, seed=seed)
            cw2 = _local_search_reduce(cw2, inc, neighbor_map, total_space, deadline)
            if len(cw2) < best_len:
                best_len = len(cw2)
                best_codewords = cw2
    
    # Final reduction
    best_codewords = _local_search_reduce(best_codewords, inc, neighbor_map, total_space, deadline)
=======
    # Phase 2: Critical word swap on the best solution
    if best_codewords is not None:
        for seed in [42, 1, 7, 13, 99, 555]:
            if time.time() >= deadline - 5.0:
                break
            cw2 = _critical_word_swap(best_codewords, inc, neighbor_map, total_space, deadline, seed=seed)
            cw2 = _local_search_reduce(cw2, inc, neighbor_map, total_space, deadline)
            if len(cw2) < best_len:
                best_len = len(cw2)
                best_codewords = cw2
    
    # Final reduction
    best_codewords = _local_search_reduce(best_codewords, inc, neighbor_map, total_space, deadline)
>>>>>>> REPLACE
