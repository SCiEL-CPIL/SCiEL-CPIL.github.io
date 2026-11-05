import itertools
import sys

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    # mask is an integer where bit i is 1 if t_subs[i] is subset of block
    block_masks = [0] * n_k_blocks
    
    # Also, precompute for each t-subset index, which blocks cover it.
    # This is needed for the swap step.
    # t_sub_to_blocks: list of lists, where t_sub_to_blocks[i] contains indices of blocks covering t_subs[i]
    t_sub_to_blocks = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            t_sub_to_blocks[idx].append(i)
        block_masks[i] = mask
        
    # 3. Greedy Set Cover for Initialization
    uncovered = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    while uncovered:
        best_idx = -1
        max_gain = 0
        
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
                
        if best_idx == -1:
            break
            
        selected_block_indices.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        
    # 4. Local Search
    # We maintain a set of selected indices for O(1) lookup
    selected_set = set(selected_block_indices)
    
    # Helper to compute counts
    def compute_counts(sel_idx_list):
        counts = [0] * n_t_subs
        for idx in sel_idx_list:
            m = block_masks[idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
        return counts

    # Helper to check if a block is redundant given counts
    def is_redundant(idx, counts):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            m ^= lsb
        return True

    # Helper to remove a block and update counts
    def remove_block(idx, counts):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            m ^= lsb

    # Helper to add a block and update counts
    def add_block(idx, counts):
        m = block_masks[idx]
        while m:
            lsb = m & -m
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            m ^= lsb

    # Main Local Search Loop
    # We iterate until no improvement is found.
    # To prevent infinite loops on swaps that don't reduce size, we can limit iterations or track visited states (hash of selected set).
    # Given the small size, a simple loop with a max iteration count is safe.
    
    max_iterations = 1000
    for _ in range(max_iterations):
        improved = False
        counts = compute_counts(selected_block_indices)
        
        # Step A: Remove redundant blocks
        redundant_blocks = []
        for idx in selected_block_indices:
            if is_redundant(idx, counts):
                redundant_blocks.append(idx)
                
        if redundant_blocks:
            # Remove one redundant block
            idx_to_remove = redundant_blocks[0]
            selected_block_indices.remove(idx_to_remove)
            selected_set.remove(idx_to_remove)
            remove_block(idx_to_remove, counts)
            improved = True
            continue # Restart loop to recalculate counts and find more redundancies
            
        # Step B: Try Swaps
        # For each block in the solution, see if we can replace it with another block
        # such that the new block covers all the "unique" subsets of the old block.
        # If so, the size remains the same, but the configuration changes.
        # This might expose new redundancies.
        
        for idx in list(selected_block_indices):
            # Find unique subsets of this block
            # These are subsets s where counts[s] == 1 and s is in block
            m = block_masks[idx]
            unique_subsets = []
            temp_m = m
            while temp_m:
                lsb = temp_m & -temp_m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_subsets.append(bit_idx)
                temp_m ^= lsb
                
            if not unique_subsets:
                # This should have been caught by redundancy check
                continue
                
            # Find a block that covers all unique_subsets
            # Intersect the lists of blocks covering each unique subset
            # Start with the list for the first unique subset
            candidate_blocks = t_sub_to_blocks[unique_subsets[0]]
            
            # Intersect with lists for other unique subsets
            for s_idx in unique_subsets[1:]:
                # Filter candidate_blocks to only those also in t_sub_to_blocks[s_idx]
                # Since lists can be large, we use a set for intersection if lists are large, 
                # but here lists are ~thousands. 
                # Optimization: if one list is much smaller, iterate it and check membership in the other.
                # t_sub_to_blocks[s_idx] is a list. 
                # Let's just do a set intersection for simplicity and speed in Python for small-medium lists.
                target_set = set(t_sub_to_blocks[s_idx])
                candidate_blocks = [b for b in candidate_blocks if b in target_set]
                
                if not candidate_blocks:
                    break
                    
            if candidate_blocks:
                # Pick a candidate block that is not currently in the solution
                # Prefer blocks that are not in selected_set
                swap_block = None
                for b in candidate_blocks:
                    if b not in selected_set:
                        swap_block = b
                        break
                        
                if swap_block is not None:
                    # Perform swap
                    # Remove idx
                    remove_block(idx, counts)
                    selected_block_indices.remove(idx)
                    selected_set.remove(idx)
                    
                    # Add swap_block
                    add_block(swap_block, counts)
                    selected_block_indices.append(swap_block)
                    selected_set.add(swap_block)
                    
                    improved = True
                    break # Restart loop after a swap to check for new redundancies
                    
        if not improved:
            break
            
    return [list(k_blocks[i]) for i in selected_block_indices]
