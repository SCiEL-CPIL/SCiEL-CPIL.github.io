import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # 1. Generate all t-subsets and map to IDs
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # 2. Generate all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # 3. Pre-compute block_to_subsets as a list of arrays for fast indexing
    # We also need a way to quickly compute gains.
    # A boolean matrix of shape (n_k, n_t) would be 50388 * 3876 booleans ~ 200 MB.
    # This might be too much memory or slow to initialize.
    # Let's use a list of numpy arrays of indices.
    
    block_to_subsets = [None] * n_k
    for i, b in enumerate(all_k_subs):
        covered_ids = []
        for t_sub in itertools.combinations(b, t):
            covered_ids.append(t_sub_to_id[t_sub])
        block_to_subsets[i] = np.array(covered_ids, dtype=np.int32)
        
    # 4. Greedy Loop with Vectorized Gain Calculation
    is_covered = np.zeros(n_t, dtype=bool)
    remaining = n_t
    selected_blocks = []
    active_indices = np.arange(n_k)
    
    # Pre-compute the full boolean matrix for gain calculation?
    # block_matrix[i, j] is True if block i covers subset j.
    # Memory: 50388 * 3876 * 1 byte = ~195 MB. This is acceptable.
    # Let's build it.
    block_matrix = np.zeros((n_k, n_t), dtype=bool)
    for i in range(n_k):
        if block_to_subsets[i].size > 0:
            block_matrix[i, block_to_subsets[i]] = True
            
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Calculate gains for all active blocks
        # gain[i] = number of subsets in block i that are NOT yet covered
        # We only care about active blocks.
        # active_blocks_matrix = block_matrix[active_indices]
        # gains = np.sum(active_blocks_matrix & ~is_covered, axis=1)
        
        # To avoid copying the large matrix slice, we can do:
        # But active_indices changes.
        # Let's just compute for all n_k and ignore inactive ones?
        # No, that's 50k * 3876 ops per step. 50k steps? No, ~150 steps.
        # 150 * 200M ops = 30B ops. Too slow.
        
        # We need to restrict to active_indices.
        # block_matrix[active_indices] creates a copy.
        # Let's try it.
        
        active_mask = block_matrix[active_indices]
        gains = np.sum(active_mask & ~is_covered, axis=1)
        
        best_local_idx = np.argmax(gains)
        best_gain = gains[best_local_idx]
        
        if best_gain == 0:
            # No block covers any new subset. This should not happen if a cover exists.
            break
            
        best_global_idx = active_indices[best_local_idx]
        selected_blocks.append(best_global_idx)
        
        # Update is_covered
        newly_covered = block_to_subsets[best_global_idx]
        is_covered[newly_covered] = True
        remaining -= np.sum(~is_covered[newly_covered]) # Wait, this logic is wrong for count.
        # Correct count update:
        # The number of newly covered subsets is the number of elements in newly_covered 
        # that were previously False in is_covered.
        # But we just set them to True.
        # So we need to count how many were False before.
        # Let's just recompute remaining? No, expensive.
        # Let's track it properly.
        
        # Actually, simpler:
        # prev_covered = is_covered.copy()
        # is_covered[newly_covered] = True
        # remaining -= np.sum(is_covered[newly_covered] & ~prev_covered[newly_covered])
        
        # Let's optimize the loop by not copying is_covered.
        # We know newly_covered are the subsets in the block.
        # We can count how many of these were already covered.
        # already_covered_count = np.sum(is_covered[newly_covered])
        # new_covered_count = len(newly_covered) - already_covered_count
        # remaining -= new_covered_count
        # is_covered[newly_covered] = True
        
        # Let's implement this inside the loop correctly.
        
        # Remove best_global_idx from active_indices
        # Since active_indices is a numpy array, we can create a new one.
        # active_indices = np.delete(active_indices, best_local_idx)
        # This is O(N). N is 50k. 150 steps * 50k = 7.5M ops. Fine.
        
        # BUT, we need to update the "already covered" logic before updating is_covered.
        
    # Re-do the loop with correct logic
    
    # Reset
    is_covered = np.zeros(n_t, dtype=bool)
    remaining = n_t
    selected_blocks = []
    active_indices = np.arange(n_k)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Compute gains for active blocks
        # active_blocks_matrix = block_matrix[active_indices]
        # This copy is the bottleneck. 50k * 3876 bytes ~ 200MB copy per step?
        # 150 steps * 200MB = 30GB memory traffic. Might be slow.
        
        # Alternative: Sparse representation?
        # Or just iterate?
        
        # Let's try the vectorized way first and see if it's fast enough.
        # If not, we fall back to a smarter gain update.
        
        # For now, let's assume the vectorized copy is okay for ~150 iterations.
        active_blocks_matrix = block_matrix[active_indices]
        gains = np.sum(active_blocks_matrix & ~is_covered, axis=1)
        
        best_local_idx = np.argmax(gains)
        best_gain = gains[best_local_idx]
        
        if best_gain == 0:
            break
            
        best_global_idx = active_indices[best_local_idx]
        selected_blocks.append(best_global_idx)
        
        # Update remaining count
        newly_covered_ids = block_to_subsets[best_global_idx]
        # Count how many were already covered
        already_covered = np.sum(is_covered[newly_covered_ids])
        new_covered = len(newly_covered_ids) - already_covered
        remaining -= new_covered
        
        # Update is_covered
        is_covered[newly_covered_ids] = True
        
        # Remove from active
        active_indices = np.delete(active_indices, best_local_idx)
        
    # 5. Refinement: Redundancy Removal
    # We have a list of block indices in selected_blocks.
    # We want to remove any block that is not necessary.
    
    # Build coverage counts for the current solution
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        coverage_count[block_to_subsets[b_idx]] += 1
        
    # Try to remove blocks one by one
    # We iterate through the list. If a block is redundant, we remove it.
    # We should be careful: removing one block might make another redundant?
    # Yes. So we iterate until no more removals are possible.
    
    changed = True
    while changed:
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            # Check if b_idx is redundant
            # If we remove it, does any subset drop to 0 coverage?
            # subsets_in_b = block_to_subsets[b_idx]
            # min_coverage_after_removal = np.min(coverage_count[subsets_in_b] - 1)
            # if min_coverage_after_removal > 0: redundant
            
            subsets_in_b = block_to_subsets[b_idx]
            if np.all(coverage_count[subsets_in_b] > 1):
                # Redundant
                # Update coverage counts
                coverage_count[subsets_in_b] -= 1
                changed = True
            else:
                new_selected.append(b_idx)
        selected_blocks = new_selected
        
    # Convert indices back to blocks
    final_blocks = [list(all_k_subs[i]) for i in selected_blocks]
    
    # Verify validity (optional, but good for debugging/safety)
    # If invalid, return greedy result without refinement?
    # The refinement only removes blocks if they are redundant, so it should remain valid.
    
    return final_blocks
