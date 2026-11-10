import itertools
import time

def construct(v, k, t):
    # Precompute t-subsets and their IDs
    t_subsets = list(itertools.combinations(range(v), t))
    t_to_id = {ts: i for i, ts in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Precompute masks for all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in k_subsets:
        mask = 0
        # For each t-subset within this block, set the bit
        for ts in itertools.combinations(block, t):
            mask |= (1 << t_to_id[ts])
        block_masks.append((mask, block))
        
    # Initial uncovered mask
    uncovered = (1 << num_t) - 1
    chosen_blocks = []
    
    # Greedy selection
    # To speed up, we might want to skip blocks that are already "covered" in some sense?
    # No, standard greedy is: pick max gain.
    
    # Optimization: If we run out of time, return what we have?
    # The task says "at most C(v,t) blocks" and "Valid when every t-subset...".
    # If we don't finish, the result is invalid.
    # However, 150s is a lot. The greedy should finish in < 10s.
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        best_mask = 0
        best_block = None
        
        for i, (mask, block) in enumerate(block_masks):
            gain = (mask & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                best_mask = mask
                best_block = block
                
        if best_gain == 0:
            # Should not happen if uncovered is not 0, unless we ran out of blocks?
            # But we have all possible blocks.
            break
            
        chosen_blocks.append(best_block)
        uncovered &= ~best_mask
        
    return chosen_blocks
