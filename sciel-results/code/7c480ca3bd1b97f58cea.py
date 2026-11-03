import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    points = list(range(v))
    
    # 1. Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # block_masks[i] is an integer where bit j is 1 if block i covers t_subset j
    block_masks = [0] * n_blocks
    
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # 3. Randomized Greedy Set Cover with multiple restarts
    best_solution = None
    best_size = float('inf')
    
    # Number of restarts based on available time
    # Precomputation takes ~2-3s, so we have ~110s for search
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Reserve 10s for final processing
            break
            
        random.seed(42 + restart)
        
        # Randomized greedy: with some probability, pick a random good block instead of the best
        uncovered_mask = (1 << n_t) - 1
        selected_block_indices = []
        
        while uncovered_mask and time.time() < deadline - 5:
            best_block_idx = -1
            best_gain = 0
            # Collect top candidates for randomization
            candidates = []
            
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [(i, gain)]
                elif gain == best_gain and gain > 0:
                    candidates.append((i, gain))
            
            if best_block_idx == -1:
                break
            
            # Randomization: with probability 0.3, pick a random candidate from top 10%
            if candidates and random.random() < 0.3:
                # Pick from top candidates
                top_k = max(1, len(candidates) // 10)
                chosen_idx, _ = random.choice(candidates[:top_k])
            else:
                chosen_idx = best_block_idx
                
            selected_block_indices.append(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
        
        # Skip if we didn't complete the cover
        if uncovered_mask:
            continue
        
        # 4. Local Search: Swap-based improvement
        # Try to swap one block for another to enable further redundancy removal
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # First, remove all redundant blocks
            counts = [0] * n_t
            for b_idx in selected_block_indices:
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
            
            # Remove redundant blocks
            new_selected = []
            for b_idx in selected_block_indices:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    improved = True
                else:
                    new_selected.append(b_idx)
            
            selected_block_indices = new_selected
            
            # Try swaps: replace one block with another that covers the same unique subsets
            if len(selected_block_indices) > 1:
                # For each block, try to find a replacement
                for i, b_idx in enumerate(selected_block_indices):
                    if time.time() > deadline - 3:
                        break
                    
                    mask = block_masks[b_idx]
                    # Find t-subsets that are only covered by this block
                    unique_to_block = []
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if counts[idx] == 1:
                            unique_to_block.append(idx)
                        m ^= lsb
                    
                    if not unique_to_block:
                        continue
                    
                    # Try to find a replacement block that covers all unique_to_block
                    # and has good overlap with other blocks
                    best_replacement = -1
                    best_score = float('inf')
                    
                    # Sample some candidate blocks for speed
                    sample_size = min(n_blocks, 5000)
                    candidates = random.sample(range(n_blocks), sample_size)
                    
                    for j in candidates:
                        if j == b_idx:
                            continue
                        if j in selected_block_indices:
                            continue
                        
                        j_mask = block_masks[j]
                        
                        # Check if j covers all unique_to_block
                        covers_all = True
                        for idx in unique_to_block:
                            if not (j_mask >> idx) & 1:
                                covers_all = False
                                break
                        if not covers_all:
                            continue
                        
                        # Score: how many of j's subsets are already covered by others
                        # (lower is better - less redundancy)
                        overlap = 0
                        m2 = j_mask
                        while m2:
                            lsb = m2 & -m2
                            idx2 = lsb.bit_length() - 1
                            if counts[idx2] >= 1:
                                overlap += 1
                            m2 ^= lsb
                        
                        # We want low overlap (blocks that add new coverage)
                        if overlap < best_score:
                            best_score = overlap
                            best_replacement = j
                    
                    if best_replacement != -1:
                        # Perform the swap
                        selected_block_indices[i] = best_replacement
                        # Update counts
                        old_mask = block_masks[b_idx]
                        new_mask = block_masks[best_replacement]
                        
                        m = old_mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        
                        m = new_mask
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                        
                        improved = True
        
        # Final redundancy removal
        if time.time() < deadline - 2:
            counts = [0] * n_t
            for b_idx in selected_block_indices:
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
            
            new_selected = []
            for b_idx in selected_block_indices:
                mask = block_masks[b_idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                else:
                    new_selected.append(b_idx)
            
            selected_block_indices = new_selected
        
        # Track best solution
        if len(selected_block_indices) < best_size:
            best_size = len(selected_block_indices)
            best_solution = list(selected_block_indices)
    
    # If no valid solution found (shouldn't happen), fallback to pure greedy
    if best_solution is None:
        uncovered_mask = (1 << n_t) - 1
        selected_block_indices = []
        while uncovered_mask and time.time() < deadline - 1:
            best_block_idx = -1
            best_gain = 0
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            if best_block_idx == -1:
                break
            selected_block_indices.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
        best_solution = selected_block_indices
    
    # 5. Convert indices back to blocks
    result_blocks = [list(all_blocks[i]) for i in best_solution]
    
    return result_blocks
