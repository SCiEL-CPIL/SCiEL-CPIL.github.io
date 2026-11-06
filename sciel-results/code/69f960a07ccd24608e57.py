import time
from itertools import combinations
import sys

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    points = range(12)
    all_blocks = list(combinations(points, 6))
    num_blocks = len(all_blocks)
    
    # Map 5-subsets to indices
    all_5_subsets = list(combinations(points, 5))
    num_5_subsets = len(all_5_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_5_subsets)}
    
    # Precompute block masks and subset to block ids
    block_masks = [0] * num_blocks
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # subset_to_block_ids[i] will contain indices of blocks that contain all_5_subsets[i]
    subset_to_block_ids = [[] for _ in range(num_5_subsets)]
    
    for i, b in enumerate(all_blocks):
        mask = 0
        for s in combinations(b, 5):
            idx = subset_to_idx[s]
            mask |= (1 << idx)
            subset_to_block_ids[idx].append(i)
        block_masks[i] = mask
        
    best_blocks = []
    
    # Symmetry breaking: Fix first block
    first_block = (0, 1, 2, 3, 4, 5)
    first_block_idx = block_to_idx[first_block]
    first_block_mask = block_masks[first_block_idx]
    
    def search(current_block_indices, covered_mask, depth):
        nonlocal best_blocks
        
        if time.time() - start_time > timeout:
            return False
        
        if len(current_block_indices) > len(best_blocks):
            best_blocks = list(current_block_indices)
            
        if len(current_block_indices) == 132:
            return True
            
        # MRV: Find the 5-subset with fewest available blocks
        min_count = float('inf')
        target_candidates = []
        
        # Iterate through all 5-subsets
        for s_idx in range(num_5_subsets):
            # Check if this 5-subset is already covered
            if (covered_mask >> s_idx) & 1:
                continue
            
            # Count available blocks containing this 5-subset
            count = 0
            candidates = []
            for b_idx in subset_to_block_ids[s_idx]:
                # Check if block is available (no conflict with covered mask)
                if not (block_masks[b_idx] & covered_mask):
                    count += 1
                    candidates.append(b_idx)
            
            if count < min_count:
                min_count = count
                target_candidates = candidates
                if count == 0:
                    return False # Dead end
                
        if not target_candidates:
            # All covered? Should not happen before 132 blocks if valid.
            # Or if we ran out of options.
            return False
            
        # Try each block in target_candidates
        for b_idx in target_candidates:
            new_covered_mask = covered_mask | block_masks[b_idx]
            if search(current_block_indices + [b_idx], new_covered_mask, depth + 1):
                return True
                
        return False
        
    # Initial call
    initial_covered = first_block_mask
    search([first_block_idx], initial_covered, 1)
    
    if len(best_blocks) < 132:
        # Fallback: If we didn't find 132, we might have found a good packing.
        # But we should probably try without fixing the first block if it fails?
        # No, symmetry breaking is safe for finding *a* solution of size 132.
        pass
        
    # Convert indices back to tuples/lists
    result = [list(all_blocks[i]) for i in best_blocks]
    return result
