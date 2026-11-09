<<<<<<< SEARCH
    def swap_search(present, max_steps=50):
        """Try to swap: remove one present triple, add two previously unsafe ones."""
        if time.time() > deadline - 5:
            return present
        
        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        # Try a few random swaps
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick a random present triple to potentially remove
            present_indices = [t for t in range(num_triples) if present[t]]
            if not present_indices:
                break
            remove_t = random.choice(present_indices)
            
            # If we remove remove_t, which unsafe triples become safe?
            # A triple t is safe after removing remove_t if all configs containing t
            # have count <= 6 after the removal.
            # Configs containing remove_t will have their count reduced by 1.
            
            newly_safe = []
            for t in range(num_triples):
                if not present[t] and unsafe_count[t] > 0:
                    # Check if all configs with count >= 7 for t are among those containing remove_t
                    all_become_safe = True
                    for ci in triple_to_configs[t]:
                        if config_count[ci] >= 7:
                            if remove_t not in config_to_triples[ci]:
                                all_become_safe = False
                                break
                    if all_become_safe:
                        newly_safe.append(t)
            
            if len(newly_safe) < 2:
                continue
            
            # Pick two to add
            add_t1, add_t2 = random.sample(newly_safe, 2)
            
            # Verify: after removing remove_t and adding add_t1, add_t2, no config exceeds 7
            # We need to check all configs
            valid = True
            for ci in range(num_configs):
                new_count = config_count[ci]
                if remove_t in config_to_triples[ci]:
                    new_count -= 1
                if add_t1 in config_to_triples[ci]:
                    new_count += 1
                if add_t2 in config_to_triples[ci]:
                    new_count += 1
                if new_count > 7:
                    valid = False
                    break
            
            if valid:
                present[remove_t] = False
                present[add_t1] = True
                present[add_t2] = True
                # Update config_count
                for ci in triple_to_configs[remove_t]:
                    config_count[ci] -= 1
                for ci in triple_to_configs[add_t1]:
                    config_count[ci] += 1
                for ci in triple_to_configs[add_t2]:
                    config_count[ci] += 1
                # Recompute unsafe_count (expensive but rare)
                unsafe_count = [0] * num_triples
                for t in range(num_triples):
                    if not present[t]:
                        for ci in triple_to_configs[t]:
                            if config_count[ci] >= 7:
                                unsafe_count[t] += 1
                break
        
        return present

    best_present = None
    best_count = 0
    
    num_restarts = 0
    while time.time() < deadline - 15:
        present = greedy_delete()
        present = local_search_add(present)
        present = swap_search(present)
        
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present
        
        num_restarts += 1
        if num_restarts > 100:
            break
    
    if best_present is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
=======
    def simulated_annealing(initial_present):
        present = initial_present[:]
        not_present_list = [t for t in range(num_triples) if not present[t]]
        present_list = [t for t in range(num_triples) if present[t]]
        
        pos_in_not_present = {t: i for i, t in enumerate(not_present_list)}
        pos_in_present = {t: i for i, t in enumerate(present_list)}

        config_count = [0] * num_configs
        for t in range(num_triples):
            if present[t]:
                for ci in triple_to_configs[t]:
                    config_count[ci] += 1
        
        unsafe_count = [0] * num_triples
        for t in range(num_triples):
            if not present[t]:
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        unsafe_count[t] += 1
        
        best_present = present[:]
        best_size = sum(present)
        current_size = best_size
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline - 5:
            move_type = random.random()
            if move_type < 0.5:
                # Try to add a triple
                if not not_present_list:
                    continue
                idx = random.randrange(len(not_present_list))
                t = not_present_list[idx]
                
                violations = 0
                for ci in triple_to_configs[t]:
                    if config_count[ci] >= 7:
                        violations += 1
                
                if violations == 0:
                    # Safe move
                    present[t] = True
                    
                    # Remove t from not_present_list
                    last = not_present_list[-1]
                    not_present_list[idx] = last
                    pos_in_not_present[last] = idx
                    not_present_list.pop()
                    del pos_in_not_present[t]
                    
                    # Add t to present_list
                    pos_in_present[t] = len(present_list)
                    present_list.append(t)
                    
                    for ci in triple_to_configs[t]:
                        old_count = config_count[ci]
                        config_count[ci] = old_count + 1
                        if old_count == 6:
                            for t_other in config_to_triples[ci]:
                                if t_other != t and not present[t_other]:
                                    unsafe_count[t_other] += 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_present = present[:]
                else:
                    if random.random() < math.exp(-violations / T):
                        present[t] = True
                        
                        last = not_present_list[-1]
                        not_present_list[idx] = last
                        pos_in_not_present[last] = idx
                        not_present_list.pop()
                        del pos_in_not_present[t]
                        
                        pos_in_present[t] = len(present_list)
                        present_list.append(t)
                        
                        for ci in triple_to_configs[t]:
                            old_count = config_count[ci]
                            config_count[ci] = old_count + 1
                            if old_count == 6:
                                for t_other in config_to_triples[ci]:
                                    if t_other != t and not present[t_other]:
                                        unsafe_count[t_other] += 1
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_present = present[:]
            else:
                # Try to remove a triple
                if not present_list:
                    continue
                idx = random.randrange(len(present_list))
                t = present_list[idx]
                
                if random.random() < math.exp(-1 / T):
                    present[t] = False
                    
                    # Remove t from present_list
                    last = present_list[-1]
                    present_list[idx] = last
                    pos_in_present[last] = idx
                    present_list.pop()
                    del pos_in_present[t]
                    
                    # Add t to not_present_list
                    pos_in_not_present[t] = len(not_present_list)
                    not_present_list.append(t)
                    
                    for ci in triple_to_configs[t]:
                        old_count = config_count[ci]
                        config_count[ci] = old_count - 1
                        if old_count == 7:
                            for t_other in config_to_triples[ci]:
                                if t_other != t and not present[t_other]:
                                    unsafe_count[t_other] -= 1
                    current_size -= 1
            
            T *= cooling_rate
        
        return best_present

    best_overall = []
    best_size_overall = 0
    
    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 15:
        # Alternate between greedy and random init
        if num_restarts % 2 == 0:
            init = greedy_delete()
            init = local_search_add(init)
        else:
            init = [False] * num_triples
            for t in range(num_triples):
                if random.random() < 0.5:
                    init[t] = True
            init = local_search_add(init)
        
        res = simulated_annealing(init)
        size = sum(res)
        if size > best_size_overall:
            best_size_overall = size
            best_overall = res
        
        num_restarts += 1

    if best_overall is None:
        return []

    return [list(all_triples[i]) for i in range(num_triples) if best_overall[i]]
>>>>>>> REPLACE
