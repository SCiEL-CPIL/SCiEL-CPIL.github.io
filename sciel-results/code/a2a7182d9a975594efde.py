import itertools
import time
import random

def construct(v, k, t):
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    block_masks = []
    for block in k_blocks:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    full_mask = (1 << num_t) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = 110.0
    
    # Run multiple restarts
    # Each restart is a greedy pass.
    # We can randomize the tie-breaking.
    
    while time.time() < deadline:
        uncovered = full_mask
        blocks_found = []
        used_blocks = set()
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            candidates = []
            
            for i, mask in enumerate(block_masks):
                if i in used_blocks:
                    continue
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_gain == -1:
                break
                
            # Randomly pick from candidates to allow different paths
            idx = random.choice(candidates)
            
            blocks_found.append(list(k_blocks[idx]))
            used_blocks.add(idx)
            uncovered &= ~block_masks[idx]
            
        if uncovered == 0: # Valid cover
            if len(blocks_found) < best_count:
                best_count = len(blocks_found)
                best_blocks = blocks_found
        else:
            # If we broke due to time, this cover is invalid.
            # We should not return it.
            pass
            
    if best_blocks is None:
        # Fallback: return the first greedy result or a valid one found
        # If no valid one found in time, we have a problem.
        # But greedy is fast, so we should find one quickly.
        # Let's do one deterministic run to guarantee a valid output if randomization fails or times out badly.
        uncovered = full_mask
        blocks_found = []
        used_blocks = set()
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i, mask in enumerate(block_masks):
                if i in used_blocks:
                    continue
                gain = (uncovered & mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            blocks_found.append(list(k_blocks[best_idx]))
            used_blocks.add(best_idx)
            uncovered &= ~block_masks[best_idx]
        return blocks_found
        
    return best_blocks
