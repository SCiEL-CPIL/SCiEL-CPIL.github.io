import itertools
import random
import time

def construct(v, k, t):
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # 2. Phase 1: Greedy Packing (Approximate MWIS on Conflict Graph)
    # Goal: Select a set of blocks (Core) such that no two blocks share t points.
    # This maximizes the efficiency of the initial coverage by avoiding redundancy.
    
    pool = set(range(num_k_subsets))
    core_blocks = []
    uncovered_mask = [1] * num_t_subsets # 1 if uncovered
    
    # Precompute list of t-subsets for each block for fast neighbor lookup
    # block_t_indices already exists.
    
    while pool:
        # Select block from pool that covers the most currently uncovered t-subsets
        best_block = -1
        max_cov = -1
        
        # Optimization: If pool is large, sampling might be faster, but 50k is manageable
        # We iterate to find the best.
        for b_idx in pool:
            cov = 0
            for t_idx in block_t_indices[b_idx]:
                if uncovered_mask[t_idx]:
                    cov += 1
            if cov > max_cov:
                max_cov = cov
                best_block = b_idx
                
        if best_block == -1:
            break
            
        core_blocks.append(best_block)
        
        # Update uncovered mask
        for t_idx in block_t_indices[best_block]:
            uncovered_mask[t_idx] = 0
            
        # Remove best_block and its "conflicts" from pool
        # Conflict: shares at least t points.
        # This is equivalent to: C is in pool if C does NOT share any t-subset with best_block.
        # So, we remove all C that share ANY t-subset with best_block.
        neighbors_to_remove = set()
        for t_idx in block_t_indices[best_block]:
            neighbors_to_remove.update(t_subset_blocks[t_idx])
        
        pool.difference_update(neighbors_to_remove)
        # Note: best_block is in neighbors_to_remove, so it gets removed too.

    # 3. Phase 2: Solve Remaining Set Cover
    # The core covers some t-subsets. We need to cover the rest.
    # Use standard Greedy Set Cover on the remaining uncovered t-subsets.
    
    # Re-initialize uncovered status for the full problem, but we know Core covers some.
    # Actually, we can just continue the greedy process on the remaining uncovered.
    
    active_blocks = set(core_blocks)
    cover_count = [0] * num_t_subsets
    
    # Calculate cover counts for core blocks
    for b_idx in active_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    # Determine which t-subsets are still uncovered
    still_uncovered = [i for i in range(num_t_subsets) if cover_count[i] == 0]
    
    if still_uncovered:
        # Greedy Set Cover for the remainder
        # We can use all blocks, or just those not in core?
        # Using all blocks allows potentially better coverage of the remainder.
        # But we must ensure we don't add duplicates if we are not careful.
        # Since we are adding to active_blocks (a set), duplicates are handled.
        
        # Gain calculation for remaining blocks
        # We want to maximize coverage of 'still_uncovered'
        
        # To speed up, we can restrict candidates to blocks that cover at least one 'still_uncovered'
        # and are not already in active_blocks?
        # Actually, a block in active_blocks might cover *more* still_uncovered?
        # No, if it was in core, it was chosen greedily. But maybe a different combination is better?
        # Standard greedy on the residual problem.
        
        residual_uncovered = set(still_uncovered)
        
        while residual_uncovered:
            best_block = -1
            max_gain = 0
            
            # Iterate over all blocks? Or just those not in active?
            # If we pick a block already in active, it adds nothing new to active_blocks set.
            # So we only need to consider blocks NOT in active_blocks.
            
            candidates = [b for b in range(num_k_subsets) if b not in active_blocks]
            
            for b_idx in candidates:
                gain = 0
                for t_idx in block_t_indices[b_idx]:
                    if t_idx in residual_uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_block = b_idx
                    
            if best_block == -1 or max_gain == 0:
                # Should not happen if cover is possible
                break
                
            active_blocks.add(best_block)
            for t_idx in block_t_indices[best_block]:
                if t_idx in residual_uncovered:
                    residual_uncovered.remove(t_idx)
                cover_count[t_idx] += 1

    # 4. Minimalization (as in current program)
    def minimize(active_blocks_set, cover_counts):
        changed = True
        while changed:
            changed = False
            items = list(active_blocks_set)
            random.shuffle(items)
            for b_idx in items:
                if b_idx not in active_blocks_set:
                    continue
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_counts[t_idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    active_blocks_set.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_counts[t_idx] -= 1
                    changed = True
        return active_blocks_set, cover_counts

    active_blocks, cover_count = minimize(active_blocks, cover_count)

    # 5. Local Search (Hill Climbing with Swaps)
    start_time = time.time()
    time_limit = 110.0
    
    while time.time() - start_time < time_limit:
        improved = False
        current_blocks = list(active_blocks)
        random.shuffle(current_blocks)
        
        for B in current_blocks:
            if B not in active_blocks:
                continue
            
            # Find t-subsets uniquely covered by B
            uncovered_by_B = []
            for t_idx in block_t_indices[B]:
                if cover_count[t_idx] == 1:
                    uncovered_by_B.append(t_idx)
            
            if not uncovered_by_B:
                continue

            # Sort by number of covering blocks to speed up intersection
            uncovered_by_B.sort(key=lambda x: len(t_subset_blocks[x]))
            
            candidates = None
            for t_idx in uncovered_by_B:
                blocks_list = t_subset_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_list)
                else:
                    candidates.intersection_update(blocks_list)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter candidates
            valid_candidates = []
            for c in candidates:
                if c != B and c not in active_blocks:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
            
            # Perform Swap
            B_new = random.choice(valid_candidates)
            
            active_blocks.remove(B)
            active_blocks.add(B_new)
            
            for t_idx in block_t_indices[B]:
                cover_count[t_idx] -= 1
            for t_idx in block_t_indices[B_new]:
                cover_count[t_idx] += 1
            
            # Minimize after swap
            active_blocks, cover_count = minimize(active_blocks, cover_count)
            improved = True
            
        if not improved:
            break

    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    return result
