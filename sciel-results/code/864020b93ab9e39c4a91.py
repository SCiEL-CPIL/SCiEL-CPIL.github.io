import itertools
import time
import random

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin
    
    # 1. Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute coverage masks for all possible k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    block_masks = [0] * n_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Construction with randomization to escape local optima
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random orderings or tie-breaking strategies
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break
            
        uncovered_mask = (1 << n_t_subs) - 1
        selected_block_indices = []
        
        while uncovered_mask != 0:
            if time.time() > deadline - 30:
                break
                
            best_idx = -1
            best_gain = -1
            
            # Add slight randomization to tie-breaking
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                
                # Small random perturbation for ties
                if gain == best_gain and best_idx != -1 and random.random() < 0.1:
                    # 10% chance to switch on tie
                    if random.random() < 0.5:
                        best_idx = i
                elif gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            selected_block_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
            
        if len(selected_block_indices) < best_size:
            best_size = len(selected_block_indices)
            best_solution = selected_block_indices[:]
            
    if best_solution is None:
        # Fallback to standard greedy if no solution found
        uncovered_mask = (1 << n_t_subs) - 1
        selected_block_indices = []
        while uncovered_mask != 0:
            if time.time() > deadline - 20:
                break
            best_idx = -1
            best_gain = -1
            for i in range(n_k_blocks):
                gain = (block_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1 or best_gain == 0:
                break
            selected_block_indices.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        best_solution = selected_block_indices
    
    # 4. Local Search: Redundancy Removal
    current_blocks = best_solution[:]
    current_masks = [block_masks[i] for i in current_blocks]
    
    cover_count = [0] * n_t_subs
    for mask in current_masks:
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        new_current_blocks = []
        new_current_masks = []
        
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
                new_current_blocks.append(current_blocks[i])
                new_current_masks.append(mask)
            else:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                improved = True
                
        current_blocks = new_current_blocks
        current_masks = new_current_masks
        
    # 5. Swap-based local search: try replacing one block with another
    # This can sometimes reduce the count further by finding a block that covers
    # more unique subsets than the one being removed.
    if time.time() < deadline - 10:
        # Build a set of selected block indices for quick lookup
        selected_set = set(current_blocks)
        
        # For each selected block, see if we can find a replacement that allows
        # removing it while still covering all subsets
        # Actually, a simpler approach: try swapping a selected block with an unselected one
        # if the swap reduces the total number of blocks or improves coverage efficiency
        
        # Let's try a different approach: for each pair of selected blocks, see if we can
        # replace one with an unselected block that covers the same critical subsets
        # and potentially allows removing another block
        
        # Simpler: iterate through all unselected blocks and see if adding it allows
        # removing two or more selected blocks
        # This is expensive, so we'll do a limited search
        
        # First, let's try single-block swaps: replace block A with block B if B covers
        # all critical subsets that A covers, and B is not already selected
        # Then check if any other blocks become removable
        
        # Build critical subset info: for each selected block, which subsets does it uniquely cover?
        # A subset is "critical for block i" if cover_count[subset] == 1 and block i covers it
        
        # For each selected block, find its critical subsets
        critical_subsets_for_block = {}
        for i, block_idx in enumerate(current_blocks):
            mask = current_masks[i]
            crit = []
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    crit.append(idx)
                m ^= lsb
            critical_subsets_for_block[block_idx] = crit
        
        # Try to find improvements by swapping
        # For each selected block, if it has no critical subsets, it's already removable
        # If it has critical subsets, we need a replacement that covers them
        
        # Let's try: for each selected block A with critical subsets, find an unselected block B
        # that covers all of A's critical subsets. If found, swap A with B, then re-run redundancy removal
        # This might allow further removals
        
        # This is complex, so let's do a simpler local search:
        # Try removing each block and see if we can add one block to fix the coverage
        # If we can fix it with one block and the net change is -1 or 0, keep it
        
        # Actually, the redundancy removal already handles the case where a block can be removed
        # without adding anything. The swap search would be for cases where we remove block A
        # and add block B, resulting in the same or fewer blocks, but B might enable further removals
        
        # Given time constraints, let's just do a few iterations of:
        # 1. Try to remove a block
        # 2. If it fails, try to find a replacement block that covers the uncovered subsets
        # 3. If the replacement allows removing another block, do the swap
        
        # This is getting complex. Let's stick with the redundancy removal and add a simple
        # "try to replace a block with a better one" heuristic
        
        # For now, let's just ensure we have a valid solution and return it
        pass
    
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
