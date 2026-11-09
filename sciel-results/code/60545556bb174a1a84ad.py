import itertools
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a greedy set cover approach 
    with NumPy-optimized coverage counting and local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage indices for every possible k-block
    # For each block, store the array of indices of the t-subsets it covers
    all_blocks = []
    block_coverage_indices = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        # Get all t-subsets of this block
        t_subs = list(itertools.combinations(block, t))
        # Map to indices
        idxs = np.array([t_subset_to_idx[ts] for ts in t_subs], dtype=np.int32)
        block_coverage_indices.append(idxs)
        
    # 3. Greedy Set Cover
    coverage_counts = np.zeros(num_t_subsets, dtype=np.int32)
    selected_block_indices = []
    
    # Track remaining uncovered count to stop early
    remaining_uncovered = num_t_subsets
    
    while remaining_uncovered > 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        # Find the block that covers the most currently uncovered t-subsets
        # We can skip blocks that are already selected, but since we're adding 
        # to a set, we just pick the best from all. 
        # Optimization: We can't easily skip selected ones in a vectorized way 
        # without a mask, but the number of blocks is ~50k, so a loop is fine.
        
        for i in range(len(all_blocks)):
            if i in selected_block_indices:
                continue
                
            idxs = block_coverage_indices[i]
            # Count how many of these indices have coverage 0
            gain = np.sum(coverage_counts[idxs] == 0)
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1 or best_gain == 0:
            # No improvement possible or no block found
            break
            
        # Select the block
        selected_block_indices.append(best_block_idx)
        
        # Update coverage counts
        idxs = block_coverage_indices[best_block_idx]
        # Count how many were 0 before incrementing to update remaining_uncovered
        newly_covered = np.sum(coverage_counts[idxs] == 0)
        remaining_uncovered -= newly_covered
        
        # Increment counts
        coverage_counts[idxs] += 1
        
    # 4. Local Search Refinement: Try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by at least one other selected block
    # i.e., coverage_count > 1 for all its subsets.
    
    if time.time() < deadline:
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            # Iterate through a copy of the list to allow removal
            for i in range(len(selected_block_indices)):
                if time.time() > deadline:
                    break
                
                block_idx = selected_block_indices[i]
                idxs = block_coverage_indices[block_idx]
                
                # Check if all subsets covered by this block have count > 1
                # If so, removing it won't uncover anything
                if np.all(coverage_counts[idxs] > 1):
                    # Remove this block
                    selected_block_indices.pop(i)
                    coverage_counts[idxs] -= 1
                    removed_any = True
                    # Decrement i to re-check the same position in the list 
                    # (which now holds the next element)
                    i -= 1
                    
    # Reconstruct chosen blocks
    chosen_blocks = [list(all_blocks[i]) for i in selected_block_indices]
    
    return chosen_blocks
