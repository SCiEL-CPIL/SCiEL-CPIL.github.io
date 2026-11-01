import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask of t-subsets covered by k-subset i
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count()

    # Helper to get list of covered t-indices from a mask
    def mask_to_list(mask):
        lst = []
        m = mask
        while m:
            lsb = m & -m
            lst.append(lsb.bit_length() - 1)
            m ^= lsb
        return lst

    best_blocks = None
    best_count = float('inf')

    def greedy_construct(seed_val):
        random.seed(seed_val)
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # Initial gains
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i])
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            # Find max gain, with random tie-breaking
            max_gain = -1
            best_cand = -1
            # Optimization: We can skip blocks that are already used.
            # Since we don't have a fast way to query "unused" efficiently in a list,
            # we just iterate. 75k iterations is fast enough.
            
            # To add randomness, we can add small noise to gain for tie-breaking or selection
            # But simple random choice among max is better.
            
            max_gain_val = -1
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain_val:
                    max_gain_val = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain_val and max_gain_val > 0:
                    candidates.append(i)
            
            if not candidates:
                return current_blocks, False
                
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0

        return current_blocks, True

    def prune_redundant(block_indices):
        if not block_indices:
            return []
        
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            m = masks[idx]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                m = masks[idx]
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    m = masks[idx]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_swap(block_indices, max_time=2.0):
        """
        Local search that allows swaps: Remove A, Add B if B covers the hole left by A.
        """
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        current_set = set(current_blocks)
        
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            m = masks[idx]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        start_ls_time = time.time()
        improved = True
        while improved and (time.time() - start_ls_time) < max_time and time.time() < deadline:
            improved = False
            random.shuffle(current_blocks)
            
            for idx_rem in current_blocks[:]:
                if time.time() > deadline:
                    return current_blocks
                
                # Check if idx_rem is removable directly
                m = masks[idx_rem]
                has_critical = False
                uncovered_by_rem = []
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        has_critical = True
                        uncovered_by_rem.append(j)
                    m ^= lsb
                
                if not has_critical:
                    # Direct removal
                    m = masks[idx_rem]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    current_blocks.remove(idx_rem)
                    current_set.discard(idx_rem)
                    improved = True
                    continue
                
                # Need to find a replacement block idx_add
                # idx_add must cover all t in uncovered_by_rem
                # And idx_add not in current_set
                
                # Heuristic: Pick a random t from uncovered_by_rem to start search
                # Or pick the one with fewest candidates?
                # Let's just pick the first one for speed, but iterate candidates.
                
                # To speed up, we can check if any block in t_to_k[uncovered_by_rem[0]] covers all others.
                target_t = uncovered_by_rem[0]
                candidates = t_to_k[target_t]
                
                found_add = -1
                for idx_cand in candidates:
                    if idx_cand in current_set:
                        continue
                    # Check if idx_cand covers all t in uncovered_by_rem
                    mask_cand = masks[idx_cand]
                    covers_all = True
                    for t_j in uncovered_by_rem:
                        if not (mask_cand & (1 << t_j)):
                            covers_all = False
                            break
                    if covers_all:
                        found_add = idx_cand
                        break
                
                if found_add != -1:
                    # Perform Swap
                    # Remove idx_rem
                    m = masks[idx_rem]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    current_blocks.remove(idx_rem)
                    current_set.discard(idx_rem)
                    
                    # Add found_add
                    m = masks[found_add]
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] += 1
                        m ^= lsb
                    current_blocks.append(found_add)
                    current_set.add(found_add)
                    
                    improved = True
                    # Note: We don't immediately prune, the next iteration will handle it.
                    # But adding a block might make other blocks redundant.
        
        return current_blocks

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
                
    if best_blocks is None:
        # Fallback
        blocks, valid = greedy_construct(0)
        if valid:
            pruned = prune_redundant(blocks)
            best_blocks = [list(k_subsets[i]) for i in pruned]
        else:
            best_blocks = []

    return best_blocks
