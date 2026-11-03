import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # Initialize uncovered mask
    uncovered_mask = (1 << n_t_subsets) - 1
    
    # Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    
    # Precompute the bitmask for each k-subset and the list of subset indices
    block_masks = []
    block_subsets = [] # List of lists of indices
    for block in k_subsets:
        mask = 0
        subs = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks.append(mask)
        block_subsets.append(subs)
    
    blocks = []
    block_indices_in_solution = []
    cover_count = [0] * n_t_subsets
    
    # Greedy selection
    while uncovered_mask:
        if time.time() > deadline:
            break
        
        best_gain = -1
        best_block_idx = -1
        best_mask = 0
        current_uncovered_count = uncovered_mask.bit_count()
        
        for i in range(len(block_masks)):
            mask = block_masks[i]
            covered = mask & uncovered_mask
            gain = covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                best_mask = covered
                if gain == current_uncovered_count:
                    break
        
        if best_gain == 0:
            break
        
        # Add block
        blocks.append(list(k_subsets[best_block_idx]))
        block_indices_in_solution.append(best_block_idx)
        
        # Update cover counts
        for idx in block_subsets[best_block_idx]:
            cover_count[idx] += 1
            
        # Update uncovered mask
        uncovered_mask &= ~best_mask
        
    # Post-processing: Remove redundant blocks
    # We do a few passes to ensure we catch all redundancies
    for _ in range(5):
        if time.time() > deadline:
            break
        new_blocks = []
        new_indices = []
        for i in range(len(blocks)):
            idx = block_indices_in_solution[i]
            # Check if this block is essential
            essential = False
            for sub_idx in block_subsets[idx]:
                if cover_count[sub_idx] == 1:
                    essential = True
                    break
            
            if essential:
                new_blocks.append(blocks[i])
                new_indices.append(idx)
            else:
                # Remove block: decrement cover counts
                for sub_idx in block_subsets[idx]:
                    cover_count[sub_idx] -= 1
        
        blocks = new_blocks
        block_indices_in_solution = new_indices
        
    return blocks
