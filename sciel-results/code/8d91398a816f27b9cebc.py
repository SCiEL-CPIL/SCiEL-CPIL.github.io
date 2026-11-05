import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Pre-compute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Pre-compute all k-blocks and their coverage masks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(all_k_blocks)
    
    # mask[i] is an integer where bit j is 1 if all_t_subsets[j] is contained in all_k_blocks[i]
    block_masks = [0] * n_k
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        # A t-subset is contained in a k-block if it is a subset of the k-block
        # We can iterate over all t-subsets of the block
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Initial state
    uncovered_mask = (1 << n_t) - 1
    counts = [0] * n_t
    solution_indices = []
    
    # Greedy Phase
    while uncovered_mask:
        if time.time() > deadline:
            break
            
        best_block_idx = -1
        best_gain = 0
        
        # Find the block that covers the most uncovered t-subsets
        for i in range(n_k):
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
            # Optional: early break if gain is maximum possible? 
            # Max gain for a block is binom(k, t). 
            # If best_gain == binom(k, t), we can break? 
            # Not necessarily, another block might have same gain but different subsets.
            # But if we just want *a* max gain, we can break if we find one with theoretical max.
            # However, tie-breaking matters. Let's just scan all.
            
        if best_gain == 0:
            # Should not happen if a cover exists, but prevents infinite loop
            break
            
        # Add the best block
        solution_indices.append(best_block_idx)
        mask = block_masks[best_block_idx]
        uncovered_mask &= ~mask
        
        # Update counts
        # Iterate through set bits in mask
        m = mask
        while m:
            # Get index of lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    # Redundancy Removal Phase
    # We want to remove blocks that don't uniquely cover any subset
    # We iterate in reverse order of addition, as later blocks are more likely to be redundant
    # Actually, any order works if we check correctly.
    
    # To speed up checking "does block i cover any subset with count==1",
    # we can just iterate the set bits of its mask.
    
    # We need to be careful when removing a block: we must decrement counts.
    # If we remove a block, other blocks might become redundant.
    # So we loop until no more blocks can be removed.
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate in reverse to remove newer blocks first?
        # Or just iterate all.
        for i in range(len(solution_indices) - 1, -1, -1):
            block_idx = solution_indices[i]
            mask = block_masks[block_idx]
            
            # Check if this block covers any subset with count == 1
            is_redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] == 1:
                    is_redundant = False
                    break
                m ^= lsb
                
            if is_redundant:
                # Remove this block
                # Decrement counts
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                    
                solution_indices.pop(i)
                changed = True
                
    # Convert indices back to blocks
    blocks = [list(all_k_blocks[i]) for i in solution_indices]
    
    return blocks
