<<<<<<< SEARCH
def _greedy_pass(v, k, t, rng, time_limit=120.0, start_time=None, tsub_to_idx=None, idx_to_tsub=None):
    """One greedy pass: repeatedly pick a t-subset to cover, build a block greedily."""
    if start_time is None:
        start_time = time.time()
    
    if tsub_to_idx is None or idx_to_tsub is None:
        tsub_to_idx, idx_to_tsub = _precompute_tsub_indices(v, t)
    
    num_tsubs = len(idx_to_tsub)
    uncovered = np.ones(num_tsubs, dtype=bool)
    blocks = []
    all_points = set(range(v))
    
    while uncovered.any():
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Pick a random uncovered t-subset
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        target_idx = uncov_indices[rng.randint(0, len(uncov_indices) - 1)]
        target = idx_to_tsub[target_idx]
        block = list(target)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Find the point that covers the most uncovered t-subsets
            best_x = None
            best_gain = -1
            candidates = all_points - block_set
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in tsub_to_idx:
                        idx = tsub_to_idx[new_tsub]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                available = list(all_points - block_set)
                if available:
                    best_x = rng.choice(available)
                else:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            for tsub in itertools.combinations(block, t):
                if tsub in tsub_to_idx:
                    uncovered[tsub_to_idx[tsub]] = False
    
    return blocks, uncovered
=======
def _greedy_pass(v, k, t, rng, time_limit=120.0, start_time=None, tsub_to_idx=None, idx_to_tsub=None):
    """One greedy pass with dual-weighted t-subset selection."""
    if start_time is None:
        start_time = time.time()
    
    if tsub_to_idx is None or idx_to_tsub is None:
        tsub_to_idx, idx_to_tsub = _precompute_tsub_indices(v, t)
    
    num_tsubs = len(idx_to_tsub)
    uncovered = np.ones(num_tsubs, dtype=bool)
    blocks = []
    all_points = set(range(v))
    
    # Dual weights for t-subsets (start at 1.0, increase for uncovered)
    dual_weights = np.ones(num_tsubs, dtype=np.float64)
    
    while uncovered.any():
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Pick an uncovered t-subset weighted by dual weight
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        
        # Weighted random selection: higher dual weight = more likely to be picked
        weights = dual_weights[uncov_indices]
        total_w = weights.sum()
        if total_w <= 0:
            target_idx = uncov_indices[rng.randint(0, len(uncov_indices) - 1)]
        else:
            probs = weights / total_w
            target_idx = uncov_indices[rng.choice(len(uncov_indices), p=probs)]
        
        target = idx_to_tsub[target_idx]
        block = list(target)
        block_set = set(block)
        
        # Build block greedily, maximizing weighted gain
        while len(block) < k:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Find the point that maximizes weighted gain
            best_x = None
            best_gain = -1
            candidates = all_points - block_set
            for x in candidates:
                gain = 0.0
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in tsub_to_idx:
                        idx = tsub_to_idx[new_tsub]
                        if uncovered[idx]:
                            gain += dual_weights[idx]
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                available = list(all_points - block_set)
                if available:
                    best_x = rng.choice(available)
                else:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Update coverage and dual weights
            for tsub in itertools.combinations(block, t):
                if tsub in tsub_to_idx:
                    idx = tsub_to_idx[tsub]
                    if uncovered[idx]:
                        uncovered[idx] = False
                        # Decrease dual weight for now-covered t-subsets
                        dual_weights[idx] = max(0.1, dual_weights[idx] * 0.5)
            # Increase dual weights for still-uncovered t-subsets
            for idx in np.where(uncovered)[0]:
                dual_weights[idx] *= 1.2
    
    return blocks, uncovered
>>>>>>> REPLACE

