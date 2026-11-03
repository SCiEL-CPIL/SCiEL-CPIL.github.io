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
    
    # Precomputation cost: O(n_blocks * C(k,t))
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # 3. Greedy Set Cover with Randomized Ties
    # We will run multiple greedy passes and keep the best one.
    best_selected = None
    best_count = float('inf')
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0: # Leave time for local search
            break
            
        uncovered_mask = (1 << n_t) - 1
        selected_block_indices = []
        
        while uncovered_mask:
            best_block_idx = -1
            best_gain = 0
            candidates = []
            
            # Iterate through all blocks
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_block_idx == -1:
                break
            
            # Randomly select from top candidates to diversify
            if len(candidates) > 1:
                selected_idx = random.choice(candidates)
            else:
                selected_idx = best_block_idx
                
            selected_block_indices.append(selected_idx)
            uncovered_mask &= ~block_masks[selected_idx]
            
        if uncovered_mask == 0:
            # Valid cover found
            if len(selected_block_indices) < best_count:
                best_count = len(selected_block_indices)
                best_selected = selected_block_indices.copy()
                
    if best_selected is None:
        # This should not happen if we don't break early.
        # But if it does, we must return a valid cover.
        # Construct a trivial valid cover (all blocks) is too big.
        # Greedy always finds a cover. The only risk is timeout.
        # Since we removed the timeout break, it will finish.
        # If we are here, something went wrong.
        return []

    selected_block_indices = best_selected
    
    # 4. Local Search: Remove redundant blocks
    # A block is redundant if every t-subset it covers is covered by at least one other selected block.
    
    # Initialize counts
    counts = [0] * n_t
    for b_idx in selected_block_indices:
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    # Iterative removal
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
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
                # Remove this block
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
                
        selected_block_indices = new_selected
        
    # 5. Local Search: Swap Heuristic to Enable Removal
    # We try to swap blocks to create redundancy in other blocks.
    
    # Helper to check if a block is redundant given current counts
    def is_redundant(block_mask, counts):
        m = block_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] <= 1:
                return False
            m ^= lsb
        return True

    # Helper to get lost mask (subsets that would become uncovered)
    def get_lost_mask(block_mask, counts):
        lost = 0
        m = block_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if counts[idx] == 1:
                lost |= (1 << idx)
            m ^= lsb
        return lost

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # First, remove any redundant blocks
        new_selected = []
        for b_idx in selected_block_indices:
            mask = block_masks[b_idx]
            if is_redundant(mask, counts):
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
        
        # If we removed something, continue loop
        if improved:
            continue

        # Try swaps
        # Iterate through blocks in reverse to handle indices safely? 
        # Or just collect swaps and apply at end? 
        # Applying one at a time is safer for state consistency.
        
        for i in range(len(selected_block_indices)):
            if time.time() > deadline - 5.0:
                break
                
            b_idx = selected_block_indices[i]
            mask = block_masks[b_idx]
            
            # Already checked redundant above, but double check in case counts changed? 
            # No, we just checked.
            
            lost_mask = get_lost_mask(mask, counts)
            if lost_mask == 0:
                # Should have been removed
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                selected_block_indices.pop(i)
                improved = True
                break

            # Find best replacement for lost_mask
            best_swap_idx = -1
            best_swap_gain = 0
            
            # To speed up, we can prioritize blocks that cover many of the lost subsets.
            # We scan all blocks. This is expensive (N_blocks * N_selected).
            # N_selected ~ 300, N_blocks ~ 75000. 300 * 75000 = 22.5M ops per pass.
            # This might be too slow in Python if we do many passes.
            # We limit the number of swap passes or use a heuristic.
            
            # Heuristic: Only try if we have significant time left.
            if time.time() < deadline - 10.0:
                for j in range(n_blocks):
                    if j == b_idx:
                        continue
                    # Optional: Skip if j is already in selected? 
                    # Swapping with a selected block is a "move", not a "swap out". 
                    # We want to replace b_idx with j. If j is already there, we just remove b_idx? 
                    # No, if j is there, and we replace b_idx with j, we have duplicate j? 
                    # Blocks must be distinct. So we cannot swap with a block already in the set.
                    if j in selected_block_indices:
                        continue
                        
                    gain = (lost_mask & block_masks[j]).bit_count()
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_idx = j
                        
            if best_swap_idx != -1 and best_swap_gain == lost_mask.bit_count():
                # Perfect swap: covers all lost subsets
                # Perform swap
                # Remove b_idx
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                    
                # Add best_swap_idx
                new_mask = block_masks[best_swap_idx]
                m = new_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
                    
                selected_block_indices[i] = best_swap_idx
                improved = True
                break # Restart loop to check for new redundancies

    # 6. Final Redundancy Removal
    counts = [0] * n_t
    for b_idx in selected_block_indices:
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for b_idx in selected_block_indices:
            mask = block_masks[b_idx]
            if is_redundant(mask, counts):
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
        selected_block_indices = new_selected
        
    result_blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    return result_blocks
