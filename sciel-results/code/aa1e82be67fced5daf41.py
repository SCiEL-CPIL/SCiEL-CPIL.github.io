import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # 1. Precompute all t-subsets and their indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    # Also store the list of t-subset indices for each block for faster iteration if needed
    # But bitmasks are efficient for intersection.
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Helper: Greedy construction
    # weights: optional list of weights to add to gain calculation
    def greedy_construct(uncovered_mask, weights=None, time_limit=None):
        selected = []
        current_uncovered = uncovered_mask
        while current_uncovered != 0:
            if time_limit and time.time() > time_limit:
                return None
            
            best_idx = -1
            best_gain = -1
            best_score = -1.0
            
            # Iterate through all blocks
            for i in range(n_k_blocks):
                gain = (block_masks[i] & current_uncovered).bit_count()
                if gain == 0:
                    continue
                
                score = float(gain)
                if weights:
                    score += weights[i]
                
                # Small random tie-breaker to avoid deterministic cycles in ILS
                if random.random() < 0.01:
                    score += random.random() * 0.1
                
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_idx = i
                elif score == best_score and random.random() < 0.1:
                    best_idx = i
                    best_gain = gain
            
            if best_idx == -1 or best_gain == 0:
                # Should not happen if uncovered is not 0 and blocks cover everything
                break
                
            selected.append(best_idx)
            current_uncovered &= ~block_masks[best_idx]
            
        return selected if current_uncovered == 0 else None

    # Helper: Local Search (Redundancy Removal + Swaps)
    def local_search(blocks, time_limit=None):
        if not blocks:
            return blocks
            
        current_blocks = blocks[:]
        current_masks = [block_masks[i] for i in current_blocks]
        
        # Calculate cover counts
        cover_count = [0] * n_t_subs
        for mask in current_masks:
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        # Phase 1: Redundancy Removal
        improved = True
        while improved and (not time_limit or time.time() < time_limit):
            improved = False
            new_blocks = []
            new_masks = []
            
            for i in range(len(current_blocks)):
                mask = current_masks[i]
                is_critical = False
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_critical = True
                        break
                    m ^= lsb
                    
                if is_critical:
                    new_blocks.append(current_blocks[i])
                    new_masks.append(mask)
                else:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    improved = True
                    
            current_blocks = new_blocks
            current_masks = new_masks
            
        # Phase 2: Swap Local Search
        # We try to replace a block with another block that allows removing another block?
        # Simple swap: Replace block i with block j if it maintains coverage and doesn't increase size.
        # Since we just removed redundancies, any swap that maintains coverage is neutral in size.
        # But it might enable further redundancy removal.
        
        if not time_limit or time.time() < time_limit - 5:
            selected_set = set(current_blocks)
            
            # Multiple passes
            for _pass in range(3):
                if time_limit and time.time() > time_limit - 5:
                    break
                improved_swap = False
                
                for i in range(len(current_blocks)):
                    if time_limit and time.time() > time_limit - 5:
                        break
                    
                    block_i = current_blocks[i]
                    mask_i = current_masks[i]
                    
                    # Find unique subsets covered by block_i
                    unique_mask = 0
                    m = mask_i
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_count[idx] == 1:
                            unique_mask |= lsb
                        m ^= lsb
                        
                    if unique_mask == 0:
                        # Should have been removed in Phase 1, but safety check
                        # Remove it
                        m = mask_i
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                        current_blocks.pop(i)
                        current_masks.pop(i)
                        selected_set.remove(block_i)
                        improved_swap = True
                        break # Restart loop since indices changed
                    
                    # Try to find a swap
                    # We need a block j not in selected_set such that:
                    # 1. j covers all unique subsets of i
                    # 2. j is "good" (covers many other things)
                    
                    # Optimization: Only try a sample of unselected blocks
                    # For speed, we just pick random unselected blocks
                    # Or iterate all if fast enough? 50k blocks * 150 steps is slow.
                    # Let's sample 1000 random unselected blocks.
                    
                    found_swap = -1
                    # Get list of unselected blocks (expensive to build every time)
                    # We can just pick random indices and check if they are in selected_set
                    attempts = 0
                    while attempts < 200:
                        j = random.randint(0, n_k_blocks - 1)
                        if j in selected_set:
                            attempts += 1
                            continue
                            
                        mask_j = block_masks[j]
                        if (mask_j & unique_mask) == unique_mask:
                            # Check if swap is beneficial
                            # It is always size-neutral.
                            # Heuristic: Prefer j that covers more "hard" subsets?
                            # Or just accept the first valid one?
                            # Let's accept the first valid one to save time.
                            found_swap = j
                            break
                        attempts += 1
                        
                    if found_swap != -1:
                        # Perform swap
                        # Remove i
                        m = mask_i
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] -= 1
                            m ^= lsb
                            
                        # Add j
                        mask_j = block_masks[found_swap]
                        m = mask_j
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            cover_count[idx] += 1
                            m ^= lsb
                            
                        current_blocks[i] = found_swap
                        current_masks[i] = mask_j
                        selected_set.remove(block_i)
                        selected_set.add(found_swap)
                        improved_swap = True
                        
                        # Run redundancy removal again immediately as swaps often create redundancies
                        improved = True
                        while improved and (not time_limit or time.time() < time_limit - 5):
                            improved = False
                            new_blocks = []
                            new_masks = []
                            for idx in range(len(current_blocks)):
                                mask = current_masks[idx]
                                is_critical = False
                                m = mask
                                while m:
                                    lsb = m & -m
                                    sub_idx = lsb.bit_length() - 1
                                    if cover_count[sub_idx] == 1:
                                        is_critical = True
                                        break
                                    m ^= lsb
                                if is_critical:
                                    new_blocks.append(current_blocks[idx])
                                    new_masks.append(mask)
                                else:
                                    m = mask
                                    while m:
                                        lsb = m & -m
                                        sub_idx = lsb.bit_length() - 1
                                        cover_count[sub_idx] -= 1
                                        m ^= lsb
                                    improved = True
                            current_blocks = new_blocks
                            current_masks = new_masks
                            # Update selected_set
                            old_set = set(current_blocks)
                            removed = selected_set - old_set
                            added = old_set - selected_set
                            selected_set -= removed
                            selected_set |= added
                            
                        i = 0 # Restart index loop
                        break # Restart outer loop to re-evaluate from start
                    
                if not improved_swap:
                    break
        
        return current_blocks

    # Main Loop: Iterated Local Search
    best_solution = None
    best_size = float('inf')
    
    # Initial Greedy
    initial_sol = greedy_construct((1 << n_t_subs) - 1, time_limit=deadline - 60)
    if initial_sol:
        local_sol = local_search(initial_sol, time_limit=deadline - 30)
        if local_sol:
            best_solution = local_sol
            best_size = len(local_sol)
    
    # ILS Loop
    while time.time() < deadline - 10:
        # Perturb
        if not best_solution:
            break
            
        # Remove 5-10 random blocks
        n_remove = max(1, int(len(best_solution) * 0.05))
        remove_indices = set(random.sample(range(len(best_solution)), min(n_remove, len(best_solution))))
        
        perturbed_blocks = [b for i, b in enumerate(best_solution) if i not in remove_indices]
        
        # Calculate uncovered mask
        covered_mask = 0
        for b in perturbed_blocks:
            covered_mask |= block_masks[b]
            
        uncovered_mask = ((1 << n_t_subs) - 1) & ~covered_mask
        
        # Repair with Greedy
        # Pass the current uncovered mask
        # We need to add blocks to cover uncovered_mask
        # The greedy function above starts from scratch. We need a version that adds to existing.
        
        # Let's modify greedy_construct to accept existing blocks
        def greedy_repair(existing_blocks, uncovered_mask, time_limit=None):
            selected = existing_blocks[:]
            current_uncovered = uncovered_mask
            selected_set = set(selected)
            
            while current_uncovered != 0:
                if time_limit and time.time() > time_limit:
                    return None
                
                best_idx = -1
                best_score = -1.0
                
                for i in range(n_k_blocks):
                    if i in selected_set:
                        continue
                    gain = (block_masks[i] & current_uncovered).bit_count()
                    if gain == 0:
                        continue
                    
                    score = float(gain)
                    # Penalty for blocks that are very similar to existing ones?
                    # Simple random tie break
                    if random.random() < 0.05:
                        score += random.random() * 0.5
                        
                    if score > best_score:
                        best_score = score
                        best_idx = i
                if best_idx == -1:
                    return None
                    
                selected.append(best_idx)
                selected_set.add(best_idx)
                current_uncovered &= ~block_masks[best_idx]
                
            return selected

        repaired_sol = greedy_repair(perturbed_blocks, uncovered_mask, time_limit=deadline - 20)
        
        if repaired_sol:
            # Local Search
            improved_sol = local_search(repaired_sol, time_limit=deadline - 10)
            
            if improved_sol and len(improved_sol) < best_size:
                best_solution = improved_sol
                best_size = len(improved_sol)
                
    if best_solution is None:
        # Fallback
        best_solution = greedy_construct((1 << n_t_subs) - 1)
        
    # Final Polish: One more local search pass on the best solution
    if time.time() < deadline - 5:
        best_solution = local_search(best_solution, time_limit=deadline - 2)
        
    # Convert indices to blocks
    blocks = [list(all_k_blocks[i]) for i in best_solution]
    return blocks
