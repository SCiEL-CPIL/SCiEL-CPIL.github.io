<<<<<<< SEARCH
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with 2-for-1 swaps and perturbation."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9998
        min_T = 0.03
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            # Choose move type: 70% 1-for-0, 20% 2-for-1, 10% perturbation
            move_type = rng.random()
            
            if move_type < 0.70:
                # 1-for-0: add a vertex, remove its conflicts
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.90:
                # 2-for-1: remove one in-set vertex, try to add two non-conflicting vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 1:
                    continue
                    
                # Pick a vertex to remove (prefer high-degree in the induced subgraph)
                remove_v = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Find candidates that could be added (not in set, not conflicting with each other)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                    
                # Filter candidates that don't conflict with each other
                # and don't conflict with the remaining in_set after removing remove_v
                remaining_in_set = in_set.copy()
                remaining_in_set[remove_v] = False
                
                # Candidates must not conflict with remaining_in_set
                cand_mask = np.zeros(N4, dtype=bool)
                for u in np.where(remaining_in_set)[0]:
                    cand_mask |= conflicts[u]
                candidates = np.where(~in_set & ~cand_mask)[0]
                
                if len(candidates) < 2:
                    continue
                    
                # Try to find a pair of non-conflicting candidates
                # Sample a few pairs to keep it fast
                max_pairs_to_try = 5
                found_pair = None
                for _ in range(max_pairs_to_try):
                    i, j = rng.integers(len(candidates), size=2)
                    if i == j:
                        continue
                    ci, cj = candidates[i], candidates[j]
                    if not conflicts[ci, cj]:
                        found_pair = (ci, cj)
                        break
                
                if found_pair is None:
                    stuck_counter += 1
                    continue
                    
                ci, cj = found_pair
                
                # Calculate delta: remove 1, add 2 => net +1
                # But we need to check if ci/cj conflict with in_set after removing remove_v
                # We already filtered, so they should be fine
                delta = 1  # net gain
                
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    # Remove remove_v
                    in_set[remove_v] = False
                    for u in neighbors[remove_v]:
                        count_conflicts[u] -= 1
                    
                    # Add ci and cj
                    for v_add in [ci, cj]:
                        to_remove = np.where(conflicts[v_add] & in_set)[0]
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                        
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_add]] += 1
                    
                    current_size = in_set.sum()
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            else:
                # Perturbation: remove 3-5 random in-set vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 3:
                    continue
                    
                n_remove = min(rng.integers(3, 6), len(in_set_indices))
                to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                
                for v in to_remove:
                    in_set[v] = False
                    for u in neighbors[v]:
                        count_conflicts[u] -= 1
                
                current_size = in_set.sum()
                stuck_counter = 0
                # Increase temperature after perturbation
                T = max(T, 2.5)
            
            if stuck_counter > 1000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
=======
    def greedy_free_recovery(in_set):
        """Greedily add any vertex with zero conflicts to in_set."""
        changed = True
        while changed:
            changed = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            # Find vertices with zero conflicts among in_set
            zero_conf = not_in_set[count_conflicts[not_in_set] == 0]
            if len(zero_conf) > 0:
                for v in zero_conf:
                    in_set[v] = True
                    count_conflicts[neighbors[v]] += 1
                    changed = True
        return in_set

    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing with 2-for-1 swaps and perturbation."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9998
        min_T = 0.03
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        # Initial free recovery
        in_set = greedy_free_recovery(in_set)
        current_size = in_set.sum()
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        # Precompute conflict row references for faster access
        while time.time() < sa_deadline:
            # Choose move type: 70% 1-for-0, 20% 2-for-1, 10% perturbation
            move_type = rng.random()
            
            if move_type < 0.70:
                # 1-for-0: add a vertex, remove its conflicts
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                counts = count_conflicts[not_in_set]
                min_count = np.min(counts)
                
                min_cands = not_in_set[counts == min_count]
                v = min_cands[rng.integers(len(min_cands))]
                
                k = min_count
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            elif move_type < 0.90:
                # 2-for-1: remove one in-set vertex, try to add two non-conflicting vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 1:
                    continue
                    
                # Pick a vertex to remove (prefer high-degree in the induced subgraph)
                remove_v = in_set_indices[rng.integers(len(in_set_indices))]
                
                # Find candidates that could be added (not in set, not conflicting with each other)
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                    
                # Filter candidates that don't conflict with each other
                # and don't conflict with the remaining in_set after removing remove_v
                remaining_in_set = in_set.copy()
                remaining_in_set[remove_v] = False
                
                # Candidates must not conflict with remaining_in_set
                cand_mask = np.zeros(N4, dtype=bool)
                for u in np.where(remaining_in_set)[0]:
                    cand_mask |= conflicts[u]
                candidates = np.where(~in_set & ~cand_mask)[0]
                
                if len(candidates) < 2:
                    continue
                    
                # Try to find a pair of non-conflicting candidates
                # Sample a few pairs to keep it fast
                max_pairs_to_try = 5
                found_pair = None
                for _ in range(max_pairs_to_try):
                    i, j = rng.integers(len(candidates), size=2)
                    if i == j:
                        continue
                    ci, cj = candidates[i], candidates[j]
                    if not conflicts[ci, cj]:
                        found_pair = (ci, cj)
                        break
                
                if found_pair is None:
                    stuck_counter += 1
                    continue
                    
                ci, cj = found_pair
                
                # Calculate delta: remove 1, add 2 => net +1
                # But we need to check if ci/cj conflict with in_set after removing remove_v
                # We already filtered, so they should be fine
                delta = 1  # net gain
                
                accept = rng.random() < np.exp(delta / T)
                
                if accept:
                    # Remove remove_v
                    in_set[remove_v] = False
                    for u in neighbors[remove_v]:
                        count_conflicts[u] -= 1
                    
                    # Add ci and cj
                    for v_add in [ci, cj]:
                        to_remove = np.where(conflicts[v_add] & in_set)[0]
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            for u in to_remove:
                                count_conflicts[neighbors[u]] -= 1
                        
                        in_set[v_add] = True
                        count_conflicts[neighbors[v_add]] += 1
                    
                    current_size = in_set.sum()
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                    
            else:
                # Perturbation: remove 3-5 random in-set vertices
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) < 3:
                    continue
                    
                n_remove = min(rng.integers(3, 6), len(in_set_indices))
                to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                
                for v in to_remove:
                    in_set[v] = False
                    for u in neighbors[v]:
                        count_conflicts[u] -= 1
                
                current_size = in_set.sum()
                stuck_counter = 0
                # Increase temperature after perturbation
                T = max(T, 2.5)
            
            if stuck_counter > 1000:
                T = max(T, 3.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Final free recovery on local best
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(local_best_in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        local_best_in_set = greedy_free_recovery(local_best_in_set)
        local_best_size = local_best_in_set.sum()
        
        return local_best_in_set, local_best_size
>>>>>>> REPLACE

