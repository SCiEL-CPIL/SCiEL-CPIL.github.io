    best_in_set = np.zeros(N_B, dtype=bool)
    best_size = 0
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
            
        rng = np.random.default_rng(restart)
        in_set = np.zeros(N_B, dtype=bool)
        alive = np.ones(N_B, dtype=bool)
        
        while np.any(alive):
            if time.time() > deadline - 10:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            
            # Min degree heuristic
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            to_remove = np.unique(np.concatenate(([chosen], neighbor_lists[chosen])))
            alive[to_remove] = False
            
        # Polish
        count_conflicts = np.zeros(N_B, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbor_lists[i]] += 1
            
        improved = True
        while improved:
            improved = False
            not_in_set = np.where(~in_set)[0]
            for v in not_in_set:
                if count_conflicts[v] == 0:
                    in_set[v] = True
                    count_conflicts[neighbor_lists[v]] += 1
                    improved = True
                    
        size = int(in_set.sum())
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
