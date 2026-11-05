import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s buffer
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Initialize uncovered bitmask: all 1s
    uncovered = (1 << num_t_subsets) - 1
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    
    # 3. Pre-compute bitmask for each k-subset and static score
    block_data = []
    for block in k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        static_score = mask.bit_count()
        block_data.append((static_score, block, mask))
        
    # Sort by static score descending to break ties in greedy selection
    # This is a heuristic: blocks that cover more total t-subsets are "better"
    block_data.sort(key=lambda x: x[0], reverse=True)
    
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        max_gain = 0
        best_block = None
        best_mask = 0
        
        # Iterate through all possible blocks
        # We use the sorted order, so ties in 'gain' will be broken by the one with higher static score
        # (since we iterate in that order and only update if gain > max_gain)
        # Wait, if gain == max_gain, we don't update. So the first one with max gain is kept.
        # Since we sorted by static score desc, the first one with max gain will have the highest static score among those with that gain.
        
        for static_score, block, mask in block_data:
            # Optimization: if static_score is 0, skip? No, static_score is always > 0 for k>=t.
            gain = (uncovered & mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_block = block
                best_mask = mask
                # If max_gain is 0, we can't do better. But we need to find if any has gain > 0.
                # If we find one, we continue to see if a higher gain exists.
        
        if best_block is None:
            # No block covers any uncovered subset
            break
            
        blocks.append(list(best_block))
        uncovered &= ~best_mask
        
    return blocks
