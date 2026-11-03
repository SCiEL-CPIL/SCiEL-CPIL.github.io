<<<<<<< SEARCH
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.995
    min_temp = 0.001
    temp = initial_temp
    
    no_improve_iter = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u
        
        if removed_any:
            temp *= cooling_rate
            continue
            
        # 2. Try block replacement with simulated annealing
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            critical = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical.append(f)
            
            if not critical:
                # Block is redundant in a sense, try removing it
                delta_u = apply_block_change(old_block, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                else:
                    apply_block_change(old_block, 1)
                    uncovered -= delta_u
                temp *= cooling_rate
                continue
            
            points_needed = set()
            for fs in critical:
                points_needed.update(fs)
                
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                    new_block_set.add(x)
                        
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        delta_u_remove = apply_block_change(old_block, -1)
                        delta_u_add = apply_block_change(new_block, 1)
                        
                        new_uncovered = uncovered + delta_u_remove + delta_u_add
                        
                        if new_uncovered == 0:
                            blocks[i] = new_block
                            uncovered = new_uncovered
                            no_improve_iter = 0
                        else:
                            # Accept with probability based on temperature
                            delta = new_uncovered  # increase in uncovered is bad
                            if delta > 0 and temp > min_temp:
                                prob = math.exp(-delta / max(temp, 1e-10))
                                if rng.random() < prob:
                                    blocks[i] = new_block
                                    uncovered = new_uncovered
                                else:
                                    apply_block_change(new_block, -1)
                                    apply_block_change(old_block, 1)
                                    uncovered = uncovered
                            else:
                                apply_block_change(new_block, -1)
                                apply_block_change(old_block, 1)
                                uncovered = uncovered
        else:
            no_improve_iter += 1

        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_counts = dict(counts)
                old_uncovered = uncovered
                
                # Remove 2-3 random blocks and try to recover
                num_remove = min(rng.randint(2, 3), len(blocks) - 1)
                remove_indices = rng.sample(range(len(blocks)), num_remove)
                remove_indices.sort(reverse=True)
                
                for idx in remove_indices:
                    b = blocks[idx]
                    delta_u = apply_block_change(b, -1)
                    uncovered += delta_u
                    blocks.pop(idx)
                
                # Try to add blocks to cover uncovered sets
                unc_list = []
                for c in itertools.combinations(range(v), t):
                    f = frozenset(c)
                    if counts.get(f, 0) == 0:
                        unc_list.append(f)
                
                attempts = 0
                while unc_list and attempts < 50:
                    attempts += 1
                    seed = rng.choice(unc_list)
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t-1):
                                fs = frozenset(list(rest) + [x])
                                if counts.get(fs, 0) == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                    
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    if new_b_tuple not in blocks:
                        blocks.append(new_b_tuple)
                        delta_u_add = apply_block_change(new_b_tuple, 1)
                        uncovered += delta_u_add
                        
                        unc_list = []
                        for c in itertools.combinations(range(v), t):
                            f = frozenset(c)
                            if counts.get(f, 0) == 0:
                                unc_list.append(f)
                    
                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                    # Reset temperature partially
                    temp = max(temp * 2.0, initial_temp * 0.5)
                else:
                    blocks = old_blocks
                    counts = old_counts
                    uncovered = old_uncovered
                    no_improve_iter += 1
                    temp = max(temp * 0.8, min_temp)
            else:
                no_improve_iter += 1
                temp = max(temp * 0.9, min_temp)
        else:
            no_improve_iter += 1

        # Cool down
        temp = max(temp * cooling_rate, min_temp)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
=======
    # Simulated annealing parameters
    initial_temp = 1.0
    cooling_rate = 0.998
    min_temp = 0.001
    temp = initial_temp
    
    no_improve_iter = 0
    max_no_improve = 300
    
    # Precompute t-subset indices for all current blocks
    block_indices = [block_tsub_indices(b) for b in blocks]
    
    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for idx in block_indices[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    block_indices.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    no_improve_iter = 0
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u
        
        if removed_any:
            temp = max(temp * cooling_rate, min_temp)
            continue
            
        # 2. Try block replacement with simulated annealing
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            critical = get_critical_tsubs(old_block)
            
            if not critical:
                # Block is redundant in a sense, try removing it
                delta_u = apply_block_change(old_block, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    block_indices.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                else:
                    apply_block_change(old_block, 1)
                    uncovered -= delta_u
                temp = max(temp * cooling_rate, min_temp)
                continue
            
            # Get points from critical t-subsets
            points_needed = set()
            for idx in critical:
                points_needed.update(all_tsubs[idx])
                
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        idx = tsub_to_idx.get(fs)
                        if idx is not None:
                            c = counts[idx]
                            if c == 0:
                                gain += 10
                            elif c == 1:
                                gain += 1
                    new_block_set.add(x)
                        
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    if new_block != old_block:
                        delta_u_remove = apply_block_change(old_block, -1)
                        delta_u_add = apply_block_change(new_block, 1)
                        
                        new_uncovered = uncovered + delta_u_remove + delta_u_add
                        
                        if new_uncovered == 0:
                            blocks[i] = new_block
                            block_indices[i] = block_tsub_indices(new_block)
                            uncovered = new_uncovered
                            no_improve_iter = 0
                        else:
                            # Accept with probability based on temperature
                            delta = new_uncovered  # increase in uncovered is bad
                            if delta > 0 and temp > min_temp:
                                prob = math.exp(-delta / max(temp, 1e-10))
                                if rng.random() < prob:
                                    blocks[i] = new_block
                                    block_indices[i] = block_tsub_indices(new_block)
                                    uncovered = new_uncovered
                                else:
                                    apply_block_change(new_block, -1)
                                    apply_block_change(old_block, 1)
                                    uncovered = uncovered
                            else:
                                apply_block_change(new_block, -1)
                                apply_block_change(old_block, 1)
                                uncovered = uncovered
        else:
            no_improve_iter += 1

        # 3. Perturbation if stuck
        if no_improve_iter > max_no_improve:
            if len(blocks) > 1:
                old_blocks = list(blocks)
                old_block_indices = [list(bi) for bi in block_indices]
                old_counts = list(counts)
                old_uncovered = uncovered
                
                # Remove 2-3 random blocks and try to recover
                num_remove = min(rng.randint(2, 3), len(blocks) - 1)
                remove_indices = rng.sample(range(len(blocks)), num_remove)
                remove_indices.sort(reverse=True)
                
                for idx in remove_indices:
                    b = blocks[idx]
                    delta_u = apply_block_change(b, -1)
                    uncovered += delta_u
                    blocks.pop(idx)
                    block_indices.pop(idx)
                
                # Try to add blocks to cover uncovered sets
                unc_list = get_uncovered_tsubs()
                
                attempts = 0
                while unc_list and attempts < 80:
                    attempts += 1
                    seed_idx = rng.choice(unc_list)
                    seed = all_tsubs[seed_idx]
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t-1):
                                fs = frozenset(list(rest) + [x])
                                idx = tsub_to_idx.get(fs)
                                if idx is not None and counts[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                    
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    if new_b_tuple not in blocks:
                        blocks.append(new_b_tuple)
                        block_indices.append(block_tsub_indices(new_b_tuple))
                        delta_u_add = apply_block_change(new_b_tuple, 1)
                        uncovered += delta_u_add
                        
                        unc_list = get_uncovered_tsubs()
                    
                if uncovered == 0:
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    no_improve_iter = 0
                    # Reset temperature partially
                    temp = max(temp * 2.0, initial_temp * 0.5)
                else:
                    blocks = old_blocks
                    block_indices = old_block_indices
                    counts = old_counts
                    uncovered = old_uncovered
                    no_improve_iter += 1
                    temp = max(temp * 0.8, min_temp)
            else:
                no_improve_iter += 1
                temp = max(temp * 0.9, min_temp)
        else:
            no_improve_iter += 1

        # Cool down
        temp = max(temp * cooling_rate, min_temp)

    # Final polish: try aggressive removal
    polish_start = time.time()
    polish_limit = min(time_limit, time.time() - start_time + 5.0)
    while time.time() - start_time < polish_limit and len(blocks) > 0:
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for idx in block_indices[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    block_indices.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed = True
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u
        if not removed:
            break

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
