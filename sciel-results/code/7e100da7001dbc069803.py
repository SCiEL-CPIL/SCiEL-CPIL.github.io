<<<<<<< SEARCH
    # VNS Loop with multiple neighborhoods
    stuck_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # N0: Add 1
        new_S = try_add_one(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N1: 1-for-2
        new_S = try_swap_1_for_2(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N2: 2-for-3
        new_S = try_swap_2_for_3(S)
        if new_S:
            S = new_S
            current_size = len(S)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # If stuck, try a 1-for-1 swap to change the structure (size remains same)
        improved_structure = False
        S_list = list(S)
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        for u in S_list:
            if time.time() > deadline:
                break
            temp_S = S - {u}
            for v in candidates:
                if is_valid_add(temp_S, v):
                    # This swap is valid
                    S = temp_S | {v}
                    improved_structure = True
                    break
            if improved_structure:
                break
        
        if improved_structure:
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 15 or no_improve_count > 50:
                # Random restart from best with perturbation
                S = best_set.copy()
                stuck_count = 0
                no_improve_count = 0
                
                # Apply 1-3 random 1-for-1 swaps to perturb
                S_list = list(S)
                num_swaps = random.randint(1, 3)
                for _ in range(num_swaps):
                    if not S_list:
                        break
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if v not in S:
                        temp_S = S - {u}
                        if is_valid_add(temp_S, v):
                            S = temp_S | {v}
                            S_list = list(S)

    return list(best_set)
=======
    # VNS Loop with multiple neighborhoods using bitmasks
    stuck_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # N0: Add 1
        new_mask = try_add_one(current_mask)
        if new_mask is not None:
            current_mask = new_mask
            current_size = bin(current_mask).count('1')
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N1: 1-for-2
        new_mask = try_swap_1_for_2(current_mask)
        if new_mask is not None:
            current_mask = new_mask
            current_size = bin(current_mask).count('1')
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # N2: 2-for-3
        new_mask = try_swap_2_for_3(current_mask)
        if new_mask is not None:
            current_mask = new_mask
            current_size = bin(current_mask).count('1')
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                no_improve_count = 0
            else:
                no_improve_count += 1
            stuck_count = 0
            improved = True
            continue
        
        # If stuck, try a 1-for-1 swap to change the structure (size remains same)
        improved_structure = False
        S_list = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
        random.shuffle(S_list)
        candidates = [x for x in range(1, N + 1) if not ((current_mask >> x) & 1)]
        
        for u in S_list:
            if time.time() > deadline:
                break
            temp_mask = current_mask & ~(1 << u)
            for v in candidates:
                if is_valid_add(temp_mask, v):
                    # This swap is valid
                    current_mask = temp_mask | (1 << v)
                    improved_structure = True
                    break
            if improved_structure:
                break
        
        if improved_structure:
            stuck_count = 0
            no_improve_count += 1
        else:
            stuck_count += 1
            no_improve_count += 1
            if stuck_count > 20 or no_improve_count > 60:
                # Random restart from best with perturbation
                current_mask = best_mask
                current_size = best_size
                stuck_count = 0
                no_improve_count = 0
                
                # Apply 2-5 random 1-for-1 swaps to perturb
                S_list = [x for x in range(1, N + 1) if (current_mask >> x) & 1]
                num_swaps = random.randint(2, 5)
                for _ in range(num_swaps):
                    if not S_list:
                        break
                    u = random.choice(S_list)
                    v = random.randint(1, N)
                    if not ((current_mask >> v) & 1):
                        temp_mask = current_mask & ~(1 << u)
                        if is_valid_add(temp_mask, v):
                            current_mask = temp_mask | (1 << v)
                            S_list = [x for x in range(1, N + 1) if (current_mask >> x) & 1]

    # Convert best_mask back to list
    best_set = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    return list(best_set)
>>>>>>> REPLACE
