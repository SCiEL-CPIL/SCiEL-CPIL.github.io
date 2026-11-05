import itertools
import time

def construct():
    points = range(12)
    # Precompute all 5-subsets and map them to indices
    all_5sets = list(itertools.combinations(points, 5))
    n_5sets = len(all_5sets)
    five_set_index = {s: i for i, s in enumerate(all_5sets)}
    
    # Precompute all 6-subsets (blocks)
    all_6sets = list(itertools.combinations(points, 6))
    n_6sets = len(all_6sets)
    
    # For each block, compute the bitmask of 5-subsets it covers
    # And for each 5-set, list the blocks that cover it
    block_masks = [0] * n_6sets
    blocks_by_5set = [[] for _ in range(n_5sets)]
    
    for i, block in enumerate(all_6sets):
        mask = 0
        for j in range(6):
            # Remove j-th element to get a 5-set
            subset = tuple(block[:j] + block[j+1:])
            idx = five_set_index[subset]
            mask |= (1 << idx)
            blocks_by_5set[idx].append(i)
        block_masks[i] = mask
        
    best_blocks = []
    target_size = 132
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute bit_count for masks? No, dynamic.
    # Use a fast popcount function
    def popcount(x):
        # Python 3.8+ has int.bit_count()
        try:
            return x.bit_count()
        except AttributeError:
            return bin(x).count('1')

    def backtrack(covered_mask, current_blocks):
        nonlocal best_blocks
        if len(current_blocks) == target_size:
            best_blocks = list(current_blocks)
            return True
            
        if time.time() > deadline:
            return False
            
        # Pruning: Check if enough uncovered 5-sets remain
        # Each new block covers 6 new 5-sets.
        # If uncovered > remaining_blocks * 6, impossible.
        uncovered_count = n_5sets - popcount(covered_mask)
        remaining_blocks = target_size - len(current_blocks)
        if uncovered_count > remaining_blocks * 6:
            return False

        available = (~covered_mask) & ((1 << n_5sets) - 1)
        if available == 0:
            return len(current_blocks) == target_size

        # MRV: Find the uncovered 5-set with minimum number of compatible blocks
        # Sample up to 20 uncovered 5-sets
        sample_indices = []
        temp = available
        max_samples = 20
        while temp and len(sample_indices) < max_samples:
            lowest_bit = temp & -temp
            idx = lowest_bit.bit_length() - 1
            sample_indices.append(idx)
            temp ^= lowest_bit
        
        if not sample_indices:
            return False
            
        # For each sampled 5-set, count compatible blocks
        best_idx = -1
        best_count = float('inf')
        best_candidates = []
        
        for idx_5set in sample_indices:
            candidates = blocks_by_5set[idx_5set]
            count = 0
            for b_idx in candidates:
                mask = block_masks[b_idx]
                if not (mask & covered_mask):
                    count += 1
            if count < best_count:
                best_count = count
                best_idx = idx_5set
                best_candidates = [b_idx for b_idx in candidates if not (block_masks[b_idx] & covered_mask)]
        
        if best_count == 0:
            return False
            
        # Iterate candidates in original order (lexicographic)
        # This is faster than scoring.
        for b_idx in best_candidates:
            mask = block_masks[b_idx]
            current_blocks.append(b_idx)
            if backtrack(covered_mask | mask, current_blocks):
                return True
            current_blocks.pop()
            
        return False

    # Symmetry breaking: The first block must contain {0, 1, 2, 3, 4}.
    for x in range(5, 12):
        first_block_tuple = (0, 1, 2, 3, 4, x)
        first_block_idx = all_6sets.index(first_block_tuple)
        
        initial_mask = block_masks[first_block_idx]
        current_blocks = [first_block_idx]
        
        if backtrack(initial_mask, current_blocks):
            result = [list(all_6sets[i]) for i in best_blocks]
            return result
            
    if best_blocks:
        return [list(all_6sets[i]) for i in best_blocks]
    return []
