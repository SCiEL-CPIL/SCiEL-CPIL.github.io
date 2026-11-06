    best_set = set()
    best_size = 0
    
    start = time.time()
    deadline = start + 115 # 5s safety margin
    
    # Phase 1: Random Greedy Restarts
    # Increase restarts if fast
    num_restarts = 2000
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s for local search
            break
        s = greedy_random()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s.copy()
            
    # Phase 2: Local Search on the best set
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 1:
            improved_set = local_search(best_set, remaining_time)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
            
    # Convert to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = all_edges[e_idx]
        result.append([u, v])
        
    return result
