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
    
    # For each block, compute the bitmask of 5-subsets it covers
    # And for each 5-set, list the blocks that cover it
    block_masks = [0] * len(all_6sets)
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
    
    # Symmetry breaking: The first block must contain {0, 1, 2, 3, 4}.
    # There are 7 such blocks: {0,1,2,3,4,x} for x in 5..11
    # Only one of these is in the Witt design.
    
    start_time = time.time()
    
    def backtrack(covered_mask, current_blocks, block_indices_remaining):
        nonlocal best_blocks
        if len(current_blocks) == target_size:
            best_blocks = list(current_blocks)
            return True
            
        if time.time() - start_time > 110: # Safety margin
            return False
            
        # Find an uncovered 5-set to force a choice
        # We can pick the first uncovered 5-set
        # To speed up, we can pick the one with fewest available blocks (MRV heuristic)
        
        # Optimization: If we just picked a block, many 5-sets are covered.
        # Let's find any uncovered 5-set.
        # Using bit operations to find the first set bit in the complement of covered_mask
        # covered_mask has 1s where covered. We want 0s.
        # ~covered_mask & ((1 << n_5sets) - 1)
        
        available = (~covered_mask) & ((1 << n_5sets) - 1)
        if available == 0:
            return False # All covered, but we didn't reach target? 
                         # Actually if all 792 are covered, we have 792/6 = 132 blocks.
                         # So if available == 0, we should have 132 blocks.
                         # If len < 132, it's a contradiction (impossible to cover all with fewer blocks? 
                         # No, if available == 0, it means all 5-sets are covered. 
                         # Since each block covers 6, and they are disjoint, 
                         # number of blocks = 792 / 6 = 132.
                         # So if available == 0, len(current_blocks) MUST be 132.
                         # If it's not, something is wrong, but logically it should be.
            return len(current_blocks) == target_size

        # Pick the 5-set with the minimum number of compatible blocks (MRV)
        # To avoid O(n_5sets) scan every time, we can just pick the first one for simplicity 
        # given the small search depth.
        # But MRV is better.
        
        # For now, pick the first uncovered 5-set
        # Find index of least significant set bit in 'available'
        # available & -available gives the lowest set bit
        lowest_bit = available & -available
        idx_5set = lowest_bit.bit_length() - 1
        
        candidates = blocks_by_5set[idx_5set]
        
        # Filter candidates that are compatible with current covered_mask
        # And not already used? 
        # We don't track used blocks explicitly, but if a block is compatible, 
        # its 5-sets are disjoint from covered. 
        # Since we pick a 5-set that is uncovered, any block covering it 
        # must have all its other 5-sets also uncovered? 
        # Yes, because if any other 5-set in the block was covered, 
        # the block would intersect covered_mask.
        
        for b_idx in candidates:
            mask = block_masks[b_idx]
            if mask & covered_mask:
                continue # Conflict
            
            # Try this block
            current_blocks.append(b_idx)
            if backtrack(covered_mask | mask, current_blocks, None):
                return True
            current_blocks.pop()
            
        return False

    # Iterate through the 7 possible first blocks
    for x in range(5, 12):
        first_block_tuple = (0, 1, 2, 3, 4, x)
        # Find index of this block
        # It's in all_6sets
        # We can just find it
        first_block_idx = all_6sets.index(first_block_tuple)
        
        initial_mask = block_masks[first_block_idx]
        current_blocks = [first_block_idx]
        
        if backtrack(initial_mask, current_blocks, None):
            # Convert indices back to lists
            result = [list(all_6sets[i]) for i in best_blocks]
            return result
            
    # Fallback: return whatever we have (should not happen)
    return [list(all_6sets[i]) for i in best_blocks]

# EVOLVE-BLOCK-END
