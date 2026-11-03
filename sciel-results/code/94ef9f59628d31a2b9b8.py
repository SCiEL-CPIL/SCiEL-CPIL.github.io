def construct(v, k, t):
    start_time = time.time()
    deadline = 140.0
    
    subs, idx_map = _precompute_subsets(v, t)
    n_subs = len(subs)
    blocks, block_masks, block_indices = _precompute_block_masks(v, k, t, subs, idx_map)
    n_blocks = len(blocks)
    
    # Precompute subset -> blocks mapping for faster replacement search
    # This is memory heavy but worth it for speed in local search
    subset_to_blocks = [[] for _ in range(n_subs)]
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        # Iterate set bits
        m = mask
        while m:
            # Get lowest set bit index
            # Python 3.11+ has int.bit_count, but for index we need trick
            # Use bin(m).index('1')? Slow.
            # Use (m & -m).bit_length() - 1
            lsb = m & -m
            s_idx = lsb.bit_length() - 1
            subset_to_blocks[s_idx].append(b_idx)
            m ^= lsb

    best_blocks = None
    best_count = float('inf')

    def indices_to_blocks(indices):
        return [blocks[i] for i in indices]

    def greedy_construct(rng, time_budget):
        uncovered = (1 << n_subs) - 1
        current_indices = []
        
        while uncovered:
            if time.time() - start_time > time_budget:
                break
            
            best_gain = -1
            best_idx = -1
            # Scan all blocks
            for idx in range(n_blocks):
                gain = _popcount(block_masks[idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            current_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        return current_indices

    def remove_redundant(indices, time_budget):
        indices = list(indices)
        n = len(indices)
        if n == 0:
            return indices
        
        changed = True
        while changed and time.time() - start_time < time_budget:
            changed = False
            # Calculate unique counts
            # O(N^2) bit ops
            unique_counts = [0] * n
            for i in range(n):
                mask_i = block_masks[indices[i]]
                others = 0
                for j in range(n):
                    if j != i:
                        others |= block_masks[indices[j]]
                unique = mask_i & ~others
                unique_counts[i] = _popcount(unique)
                
            # Remove first block with 0 unique
            for i in range(n):
                if unique_counts[i] == 0:
                    indices.pop(i)
                    changed = True
                    break
            
        return indices

    def local_search(indices, rng, time_budget):
        indices = list(indices)
        n = len(indices)
        
        while time.time() - start_time < time_budget and n > 0:
            i = rng.randint(0, n - 1)
            removed_idx = indices[i]
            removed_mask = block_masks[removed_idx]
            
            # Compute remaining mask
            remaining_mask = 0
            for j in range(n):
                if j != i:
                    remaining_mask |= block_masks[indices[j]]
            
            deficit = removed_mask & ~remaining_mask
            if deficit == 0:
                indices.pop(i)
                n -= 1
                continue
            
            # Find replacement
            # Find the subset in deficit with fewest covering blocks
            # Get list of set bits in deficit
            deficit_subsets = []
            m = deficit
            while m:
                lsb = m & -m
                s_idx = lsb.bit_length() - 1
                deficit_subsets.append(s_idx)
                m ^= lsb
            
            if not deficit_subsets:
                continue
                
            # Pick the rarest subset
            # Sort by len(subset_to_blocks[s_idx])
            deficit_subsets.sort(key=lambda s: len(subset_to_blocks[s]))
            rarest = deficit_subsets[0]
            
            # Check candidates from the rarest subset
            candidates = subset_to_blocks[rarest]
            rng.shuffle(candidates)
            
            found = False
            for cand in candidates:
                if cand == removed_idx:
                    continue
                if cand in indices:
                    continue
                # Check if cand covers all deficit
                if deficit & ~block_masks[cand] == 0:
                    indices[i] = cand
                    found = True
                    break
            
            if not found:
                # If we can't find a single block, this swap is bad.
                # We could try adding 2 blocks? That increases size.
                # We only want to swap 1 for 1 to maintain size and enable removals.
                pass

        return indices

    num_trials = 15
    per_trial_time = 8.0
    
    for trial in range(num_trials):
        if time.time() - start_time > deadline - 10:
            break
            
        rng = random.Random(42 + trial)
        
        g_indices = greedy_construct(rng, per_trial_time / 2)
        if not g_indices:
            continue
        
        g_indices = remove_redundant(g_indices, per_trial_time / 4)
        
        l_indices = local_search(g_indices, rng, per_trial_time / 4)
        
        l_indices = remove_redundant(l_indices, 2.0)
        
        count = len(l_indices)
        if count < best_count:
            best_count = count
            best_blocks = indices_to_blocks(l_indices)
            
    if best_blocks is None:
        # Fallback
        uncovered = (1 << n_subs) - 1
        best_indices = []
        while uncovered:
            best_gain = -1
            best_idx = -1
            for idx in range(n_blocks):
                gain = _popcount(block_masks[idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx == -1: break
            best_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
        best_blocks = indices_to_blocks(best_indices)
        
    return best_blocks
