import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks and their bitmasks of covered t-subsets
    # Also store the actual points of the block for output
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_bitmasks[i] is an integer bitmask where bit j is 1 if t_subsets[j] is in all_k_blocks[i]
    # block_points[i] is the tuple of points
    block_bitmasks = [0] * n_k_blocks
    block_points = all_k_blocks
    
    # Map points to t-subsets containing them? Not needed for bitmask greedy.
    # We need to build the bitmasks.
    # For each block, find all t-subsets within it.
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask

    def greedy_cover():
        uncovered_mask = (1 << n_t_subsets) - 1
        selected_blocks = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = 0
            
            # Iterate through all blocks to find the one covering most uncovered subsets
            # Optimization: We can skip blocks that are already selected? 
            # No, a block might cover new subsets if added later? 
            # Actually, in set cover, once a set is chosen, it's fixed. 
            # But here we are constructing the set. 
            # Standard greedy: pick set with max marginal gain.
            
            for i in range(n_k_blocks):
                # If the block is already selected, gain is 0 for new coverage? 
                # Actually, we just need to cover the remaining uncovered.
                # If a block is already in selected_blocks, its contribution to 'uncovered' is 0 
                # because uncovered_mask already has 0s where it covers.
                # So we can just compute gain.
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break # Should not happen if uncovered is not empty and blocks cover universe
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
            
        return selected_blocks

    def local_search_remove(blocks_indices):
        # counts[j] = number of selected blocks covering t-subset j
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        
        for b_idx in blocks_indices:
            mask = block_bitmasks[b_idx]
            # Extract set bits
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        # Iterate until no more removals
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                mask = block_bitmasks[b_idx]
                m = mask
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                # Remove them
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    mask = block_bitmasks[b_idx]
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                changed = True
                
        return blocks_indices

    # Run Greedy
    initial_blocks = greedy_cover()
    
    # Run Local Search
    current_blocks = local_search_remove(initial_blocks)
    
    # If time remains, try randomized restarts or perturbations
    # Strategy: Randomly remove a few blocks, then re-greedy the gaps.
    # This can escape local minima where no single block is removable.
    
    best_blocks = current_blocks
    best_count = len(current_blocks)
    
    # Simple perturbation: 
    # 1. Take current_blocks
    # 2. Remove a random subset of non-essential blocks (or even essential ones, creating holes)
    # 3. Run greedy on the holes
    # 4. If result is smaller, update best.
    
    # To do this efficiently, we need a function that takes a list of blocks, 
    # removes some, and covers the rest.
    
    def cover_holes(existing_blocks, holes_mask):
        # holes_mask is a bitmask of t-subsets that are currently UNCOVERED
        # We run greedy to cover these holes.
        # Note: existing_blocks are kept.
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                # We can add any block. 
                # To avoid adding blocks already in existing_blocks (redundant), 
                # we could check, but gain will be 0 if it doesn't cover new holes.
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks

    # Perform a few perturbation attempts
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        # Create a copy of current best
        perturbed_blocks = list(best_blocks)
        
        # Randomly remove 5-10 blocks
        num_remove = min(len(perturbed_blocks), random.randint(5, 10))
        if num_remove == 0:
            break
        to_remove = random.sample(perturbed_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in perturbed_blocks if b not in remove_set]
        
        # Calculate holes
        # We need to know which t-subsets are covered by the remaining perturbed_blocks
        # And which were covered by the removed ones.
        # Actually, easier: 
        # 1. Compute coverage of perturbed_blocks.
        # 2. Holes = Universe - Coverage(perturbed_blocks)
        
        # This calculation is expensive if done from scratch.
        # Let's optimize: 
        # We have 'counts' from the last local search? No, that was for best_blocks.
        # Let's just recompute coverage for the perturbed set.
        
        # Optimization: Only do this if the potential gain is high.
        # Removing 5 blocks, we need to add < 5 blocks to improve.
        
        # Compute mask of covered subsets by perturbed_blocks
        covered_mask = 0
        for b_idx in perturbed_blocks:
            covered_mask |= block_bitmasks[b_idx]
            
        holes_mask = ((1 << n_t_subsets) - 1) & ~covered_mask
        
        # Cover holes
        added_blocks = cover_holes(perturbed_blocks, holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        if new_count < best_count:
            # Run local search on the new set to clean it up
            cleaned_blocks = local_search_remove(new_total_blocks)
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_count = len(best_blocks)
            else:
                best_blocks = new_total_blocks # Fallback if local search failed? No, keep best.
                # Actually, if local search didn't improve, the raw new_total_blocks might be better than best?
                # No, best_count was the size of best_blocks.
                # If len(new_total_blocks) < best_count, we should update best_blocks to new_total_blocks
                # even if local search didn't reduce it further? 
                # Yes.
                if len(new_total_blocks) < best_count:
                    best_blocks = new_total_blocks
                    best_count = len(best_blocks)

    # Convert indices back to point lists
    final_blocks = [list(block_points[i]) for i in best_blocks]
    
    # Ensure distinctness (should be by construction)
    # Ensure validity (checked by evaluator)
    
    return final_blocks
