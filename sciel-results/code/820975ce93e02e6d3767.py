import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    full_mask = (1 << num_t_subs) - 1
    
    # Generate all k-blocks and their masks
    k_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_blocks:
        mask = 0
        # Each k-block contains C(k, t) t-subsets
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(k_blocks)
    
    best_blocks = None
    best_count = float('inf')
    
    # Randomized Greedy Search
    # We run multiple restarts with different random seeds/orders
    # The greedy step: pick block with max gain
    # To diversify, we can randomize the order of blocks considered or break ties randomly
    
    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = full_mask
        selected = []
        
        # Optimization: Maintain a list of active blocks?
        # For now, simple scan. 75k blocks is small enough.
        
        while uncovered:
            best_gain = -1
            best_block_idx = -1
            # Shuffle indices to break ties randomly? 
            # Or just pick first max. Randomizing the start or tie-breaking helps.
            
            # To speed up, we can pre-shuffle the list of block indices once per restart?
            # No, the gain changes.
            
            # Iterate all blocks
            for i in range(num_blocks):
                gain = (block_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    if gain == num_t_subs: # Perfect cover, unlikely
                        break
                # Tie-breaking: if gain == best_gain, we could randomize
                # But strict > is faster. Let's add a small random jitter to tie-breaking
                # or just rely on the order.
            
            if best_block_idx == -1:
                # Should not happen if uncovered is not empty and blocks exist
                break
                
            selected.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]
            
            if time.time() > deadline:
                return None # Timeout in this restart
        
        # Local Removal Phase
        # Try to remove redundant blocks
        # Repeat until stable
        changed = True
        while changed:
            changed = False
            new_selected = []
            for i in selected:
                # Check if block i is redundant
                # Compute union of all other blocks
                # This is O(N^2) per pass. N is ~300. 300^2 = 90k ops.
                # 90k * (bitwise OR + compare) is fast.
                
                # Optimization: 
                # Instead of O(N^2) every time, we can just check if 
                # mask[i] is subset of union(others).
                
                # Let's do a simpler check:
                # If we remove i, does the cover break?
                # union_others = 0
                # for j in selected:
                #   if j != i: union_others |= block_masks[j]
                # if (union_others & full_mask) == full_mask:
                #    redundant
                
                # This is expensive if done naively.
                # Alternative: 
                # Count how many times each t-subset is covered?
                # If count > 1 for all subsets in block i, it's redundant.
                # Maintaining counts is complex during removal.
                
                # Let's stick to the O(N^2) check but optimize it.
                # N is small (< 500).
                
                is_redundant = True
                # We can break early if we find a subset in block i not covered by others
                # But computing "others" union is the cost.
                
                # Let's just do it.
                union_others = 0
                for j in selected:
                    if j != i:
                        union_others |= block_masks[j]
                        # Early exit if union_others covers all of block_i?
                        # if (union_others & block_masks[i]) == block_masks[i]:
                        #    break
                        # This check is cheap.
                        if (union_others & block_masks[i]) == block_masks[i]:
                            break
                
                if (union_others & block_masks[i]) != block_masks[i]:
                    is_redundant = False
                    # It is NOT redundant, keep it
                    new_selected.append(i)
                else:
                    # It IS redundant, remove it
                    changed = True
            
            selected = new_selected
            
            if time.time() > deadline:
                return None
                
        return selected

    # Run multiple restarts
    # How many? 
    # Greedy is fast. Let's try 50-100 restarts.
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Randomize the order of blocks? 
        # The greedy loop iterates 0..N-1.
        # To diversify, we can shuffle the indices of the blocks.
        # But we need to map back to the original indices for the result.
        
        # Alternative: Randomize the "tie-breaking" by adding noise to the gain?
        # Or just shuffle the `k_blocks` list?
        
        # Let's create a shuffled list of indices for the search order.
        # This changes which block is picked in case of ties or if we stop early?
        # No, greedy picks the MAX. The order only matters for ties.
        # To truly diversify, we need to perturb the objective or the candidates.
        
        # Strategy: Randomly exclude some blocks? No, might hurt.
        # Strategy: Randomly start with a random block?
        
        # Let's try: In the greedy step, if there are multiple blocks with max gain, pick one randomly.
        
        def greedy_cover_diverse(random_seed):
            random.seed(random_seed)
            uncovered = full_mask
            selected = []
            
            while uncovered:
                best_gain = -1
                candidates = []
                
                for i in range(num_blocks):
                    gain = (block_masks[i] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [i]
                    elif gain == best_gain:
                        candidates.append(i)
                
                if not candidates:
                    break
                    
                # Pick randomly from candidates
                best_block_idx = random.choice(candidates)
                
                selected.append(best_block_idx)
                uncovered &= ~block_masks[best_block_idx]
                
                if time.time() > deadline:
                    return None
            
            # Local Removal
            changed = True
            while changed:
                changed = False
                new_selected = []
                for i in selected:
                    union_others = 0
                    for j in selected:
                        if j != i:
                            union_others |= block_masks[j]
                            if (union_others & block_masks[i]) == block_masks[i]:
                                break
                    
                    if (union_others & block_masks[i]) != block_masks[i]:
                        new_selected.append(i)
                    else:
                        changed = True
                
                selected = new_selected
                if time.time() > deadline:
                    return None
                    
            return selected

        res = greedy_cover_diverse(r)
        if res is not None:
            if len(res) < best_count:
                best_count = len(res)
                best_blocks = res

    if best_blocks is None:
        # Fallback to naive if everything timed out (unlikely)
        # Or return a valid but potentially suboptimal cover
        # The greedy above should always return something unless it times out mid-way.
        # If it times out, we might have a partial list. 
        # We should ensure we return a VALID cover.
        # The greedy loop `while uncovered` ensures validity if it finishes.
        # If it breaks due to timeout, it's invalid.
        # So we must only accept completed runs.
        
        # If no run completed, we need a fallback.
        # But 110s is plenty for 100 restarts of this size.
        
        # Just in case, run a single deterministic greedy without local search?
        # No, the first restart (seed 0) is deterministic if random.choice is not used?
        # random.choice with seed 0 is deterministic.
        # So best_blocks should be set.
        
        pass

    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in best_blocks]
