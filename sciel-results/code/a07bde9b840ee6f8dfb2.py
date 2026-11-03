import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t_subs = len(t_subs)
    
    # 2. Generate all k-subsets and compute their coverage masks
    # mask[b] is an integer where bit i is 1 if t_sub i is contained in k_sub b
    k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(k_subs)
    
    # Precompute masks
    masks = [0] * n_k_subs
    for idx, k_sub in enumerate(k_subs):
        mask = 0
        # Generate all t-subsets of this k-subset
        for t_sub in itertools.combinations(k_sub, t):
            mask |= (1 << sub_to_idx[t_sub])
        masks[idx] = mask
        
    # Check time
    if time.time() > deadline:
        return [] # Fallback, shouldn't happen

    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    blocks = []
    used_blocks = set()
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        best_gain = 0
        
        # Find block that covers the most uncovered t-subsets
        for i in range(n_k_subs):
            if i in used_blocks:
                continue
            gain = (uncovered & masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
            # Early exit if we found a perfect cover for this step? 
            # Max possible gain is number of uncovered subsets.
            # But we can't easily know that without counting bits in uncovered.
            # If best_gain == uncovered.bit_count(), we can break.
            if best_gain == uncovered.bit_count():
                break
                
        if best_block_idx == -1:
            # Should not happen if covering is possible
            break
            
        blocks.append(list(k_subs[best_block_idx]))
        used_blocks.add(best_block_idx)
        uncovered &= ~masks[best_block_idx]
        
    return blocks
