import itertools
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a greedy set cover approach 
    with precomputed block coverage bitmasks, followed by vectorized local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # 2. Precompute coverage bitmask for every possible k-block
    # We will store these as numpy arrays of uint8 for fast vectorized operations later
    # But for the greedy phase, integer bitmasks are faster for intersection/gain calculation
    block_masks_int = []
    all_blocks = []
    
    # Generate all k-blocks
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks_int.append(mask)
        
    # 3. Greedy Set Cover
    current_uncovered = (1 << num_t_subsets) - 1
    
    chosen_block_indices = []
    used_block_indices = set()
    
    def count_bits(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    while current_uncovered != 0:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        
        for i in range(len(block_masks_int)):
            if i in used_block_indices:
                continue
                
            intersection = block_masks_int[i] & current_uncovered
            gain = count_bits(intersection)
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                
        if best_block_idx == -1:
            break
            
        chosen_block_indices.append(best_block_idx)
        used_block_indices.add(best_block_idx)
        
        current_uncovered &= ~block_masks_int[best_block_idx]
        
    if not chosen_block_indices:
        return []

    # 4. Vectorized Local Search Refinement
    # Convert selected block masks to a numpy boolean matrix
    # Shape: (num_chosen, num_t_subsets)
    # We can do this efficiently by unpacking bits or using a lookup table if num_t_subsets is small.
    # num_t_subsets is ~3876. 
    # Creating a boolean matrix of size (150, 3876) is ~580k booleans, which is small.
    
    num_chosen = len(chosen_block_indices)
    # Initialize matrix
    # Using uint8 to save memory and allow summing
    coverage_matrix = np.zeros((num_chosen, num_t_subsets), dtype=np.uint8)
    
    for i, b_idx in enumerate(chosen_block_indices):
        mask = block_masks_int[b_idx]
        # Efficiently set bits in the numpy array
        # We can iterate through set bits, which is ~35 per block (7C4=35)
        # This is fast enough for 150 blocks.
        m = mask
        while m:
            # Get lowest set bit index
            lsb = (m & -m).bit_length() - 1
            coverage_matrix[i, lsb] = 1
            m &= m - 1
            
    # Compute coverage counts
    coverage_count = coverage_matrix.sum(axis=0).astype(np.int32)
    
    # Iterative removal of redundant blocks
    # A block is redundant if it doesn't cover any subset with count == 1
    active_indices = list(range(num_chosen))
    
    while True:
        if time.time() > deadline:
            break
            
        if not active_indices:
            break
            
        # Get indices of subsets with count == 1
        critical_subsets = np.where(coverage_count == 1)[0]
        
        if len(critical_subsets) == 0:
            # If no subset has count 1, we can remove any block?
            # No, if we remove a block, counts drop. If min count was 2, it becomes 1.
            # So we can remove blocks one by one.
            # But actually, if min count >= 2, we can remove ALL blocks?
            # No, we must leave enough blocks to cover the subsets.
            # If we remove a block, subsets it covered drop by 1.
            # If they were 2, they become 1. They are still covered.
            # So yes, if min(count) >= 2, we can remove blocks until min(count) == 1.
            # However, the condition "block doesn't cover any critical subset" is the general rule.
            # If critical_subsets is empty, then NO block covers any critical subset.
            # So ALL blocks are redundant?
            # Wait. If critical_subsets is empty, it means all counts are >= 2 (or 0, but we only have covered ones).
            # If a block covers only subsets with count >= 2, removing it makes them >= 1.
            # So yes, if critical_subsets is empty, we can remove *any* block.
            # We can remove them all? No, if we remove all, counts become 0.
            # We can remove blocks one by one.
            # Let's just pick one (or all that are "safe")?
            # Actually, if critical_subsets is empty, every block is safe to remove *individually*.
            # We can remove all of them? No.
            # Let's just remove one block per iteration if critical_subsets is empty?
            # Or better: if critical_subsets is empty, we can remove blocks until critical_subsets is non-empty.
            # This implies we can remove `min(coverage_count) - 1` blocks?
            # Not exactly, because different blocks cover different subsets.
            # But we can simply remove one block, update, and repeat.
            # To speed up, if critical_subsets is empty, we can remove a block that covers the "least" number of subsets?
            # Or just the first one.
            
            # Let's just remove the first active block
            remove_idx = active_indices[0]
            active_indices.remove(remove_idx)
            coverage_count -= coverage_matrix[remove_idx]
            continue
            
        # Check which active blocks cover any critical subset
        # We need to check intersection of block mask with critical_subsets
        # Extract rows for active blocks
        active_mask_submatrix = coverage_matrix[np.array(active_indices)][:, critical_subsets]
        
        # For each block, does it have any 1 in the critical columns?
        covers_critical = active_mask_submatrix.any(axis=1)
        
        # Find blocks that do NOT cover any critical subset
        redundant_local_indices = np.where(~covers_critical)[0]
        
        if len(redundant_local_indices) == 0:
            break # No more blocks to remove
            
        # Remove these blocks
        # Map local indices to global active indices
        global_remove_indices = [active_indices[i] for i in redundant_local_indices]
        
        # Update coverage count
        # Sum the masks of removed blocks
        removed_masks = coverage_matrix[np.array(global_remove_indices)]
        coverage_count -= removed_masks.sum(axis=0)
        
        # Remove from active list
        for g_idx in global_remove_indices:
            active_indices.remove(g_idx)
            
    # Reconstruct final blocks
    final_blocks = [list(all_blocks[i]) for i in active_indices]
    
    return final_blocks
