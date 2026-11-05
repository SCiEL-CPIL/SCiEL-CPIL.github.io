import itertools
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    # Map t-subset (tuple) to index
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    # A block covers a t-subset if the t-subset is a subset of the block
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    # mask is an integer where bit i is 1 if t_subs[i] is subset of block
    block_masks = [0] * n_k_blocks
    
    for i, block in enumerate(k_blocks):
        mask = 0
        # Generate all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    while uncovered:
        # Use list comprehension for faster gain calculation
        gains = [ (block_masks[i] & uncovered).bit_count() for i in range(n_k_blocks) ]
        max_gain = max(gains)
        if max_gain == 0:
            break
        best_idx = gains.index(max_gain)
        
        selected_block_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search: Remove redundant blocks
    if selected_block_indices:
        counts = [0] * n_t_subs
        for idx in selected_block_indices:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
                
        final_indices = []
        for idx in selected_block_indices:
            m = block_masks[idx]
            essential = False
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    essential = True
                    break
                m ^= lsb
                
            if essential:
                final_indices.append(idx)
            else:
                m = block_masks[idx]
                while m:
                    lsb = m & -m
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    m ^= lsb
                    
        selected_block_indices = final_indices
        
    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in selected_block_indices]
