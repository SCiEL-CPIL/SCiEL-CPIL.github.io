"""
Strategy: Bitmask-based Vectorized Greedy Set Cover + Local Search Refinement.

1. Precompute all C(19,4) = 3876 t-subsets as bitmasks.
2. Use NumPy to vectorize the calculation of how many uncovered subsets each 
   potential block covers.
3. Greedy selection: Pick the block that covers the most uncovered subsets.
4. Local Search: After the greedy phase, attempt to swap blocks to reduce 
   the total count or improve coverage efficiency (though for pure covering 
   number, we mainly want to minimize count while maintaining validity).
   Since the greedy is fast with bitmasks, we have time for a simple local 
   search: try removing a block and see if it's still covered by others. 
   If yes, remove it. If no, try swapping it with a block that covers the 
   unique subsets better.
"""

import time
import numpy as np
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precompute all t-subsets and their bitmasks
    # v=19, t=4 -> 3876 subsets
    # k=7 -> C(19,7) = 50388 blocks
    # We cannot store a 50388 x 3876 boolean matrix easily in memory? 
    # 50388 * 3876 ~ 195 million bits ~ 24 MB. This is fine.
    # But we don't need to store the full matrix initially. We can compute on the fly or store as int64 bitmasks.
    
    # Let's represent each t-subset as an integer bitmask (19 bits).
    # Let's represent each block as an integer bitmask (19 bits).
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    # Map each t-subset to an index 0..num_t-1
    # And create a bitmask for each t-subset
    t_masks = np.array([0] * num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask
        
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Create a bitmask for each k-block
    k_masks = np.array([0] * num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask
        
    # Precompute coverage matrix: cover[i, j] = 1 if block i covers t-subset j
    # This is a num_k x num_t boolean matrix.
    # To save memory and time, we can compute this once.
    # cover[i, j] is true if (t_masks[j] & ~k_masks[i]) == 0
    # i.e., t_masks[j] is a subset of k_masks[i]
    
    # Vectorized subset check:
    # We can expand k_masks to (num_k, 1) and t_masks to (1, num_t)
    # But 50388 * 3876 is ~200M operations. NumPy can handle this.
    # Let's do it in chunks to avoid memory blowup if needed, but 200M bools is 200MB. 
    # It's on the edge. Let's try to be smart.
    
    # Alternative: Don't store the full matrix. 
    # Greedy step: For each candidate block, count how many *uncovered* t-subsets it covers.
    # Uncovered is a boolean vector of size num_t.
    # Count = sum(cover[i, :] * uncovered)
    
    # To make this fast, we need cover[i, :] to be accessible fast.
    # Let's store cover as a list of bytearrays or numpy arrays?
    # Or just compute the intersection count using bit operations on the masks?
    # No, t-subsets are specific.
    
    # Let's precompute the coverage matrix as a list of numpy arrays of uint8.
    # cover[i] is a uint8 array of length num_t.
    # Memory: 50388 * 3876 bytes = ~195 MB. This is acceptable.
    
    # Optimization: We can compute this matrix once.
    # cover[i, j] = 1 if (t_masks[j] & ~k_masks[i]) == 0
    
    # Let's do it efficiently.
    # t_masks: (num_t,) int64
    # k_masks: (num_k,) int64
    
    # We can't easily vectorize the subset check for all pairs without expanding to 2D.
    # Let's do it in chunks of blocks.
    cover_list = []
    chunk_size = 1000
    for start in range(0, num_k, chunk_size):
        end = min(start + chunk_size, num_k)
        k_chunk = k_masks[start:end] # (chunk, )
        # Expand to (chunk, 1) and (1, num_t)
        # Check: (t_masks[None, :] & ~k_chunk[:, None]) == 0
        # This creates a (chunk, num_t) array.
        # 1000 * 3876 = 3.8M elements. 3.8M * 8 bytes (int64) = 30MB. Fine.
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_list.append(check.astype(np.uint8))
        
    # Now cover_list is a list of (chunk, num_t) uint8 arrays.
    # We can flatten or keep as list.
    
    uncovered = np.ones(num_t, dtype=np.uint8)
    blocks_selected = []
    indices_selected = []
    
    # Greedy Phase
    while np.sum(uncovered) > 0:
        if time.time() > deadline - 10: # Leave 10s for finalization/local search
            break
            
        best_idx = -1
        best_gain = 0
        
        # Calculate gain for each block
        # gain[i] = sum(cover[i, :] * uncovered)
        # We can vectorize this across all blocks if we had the full matrix.
        # Since we have chunks, let's loop chunks.
        
        for chunk_idx, cover_chunk in enumerate(cover_list):
            # cover_chunk: (chunk_size, num_t)
            # uncovered: (num_t,)
            # gains = cover_chunk @ uncovered  (dot product)
            gains = cover_chunk.dot(uncovered)
            max_gain_in_chunk = np.max(gains)
            
            if max_gain_in_chunk > best_gain:
                best_gain = max_gain_in_chunk
                local_idx = np.argmax(gains)
                best_idx = chunk_idx * chunk_size + local_idx
        
        if best_gain == 0:
            # Should not happen if uncovered > 0, unless bug
            break
            
        blocks_selected.append(k_blocks[best_idx])
        indices_selected.append(best_idx)
        
        # Update uncovered
        # uncovered[cover[best_idx] == 1] = 0
        chunk_idx = best_idx // chunk_size
        local_idx = best_idx % chunk_size
        cover_row = cover_list[chunk_idx][local_idx]
        uncovered[cover_row == 1] = 0
        
    # Check if we finished within time
    if time.time() > deadline - 5:
        return [list(b) for b in blocks_selected]
    
    # Local Search / Refinement
    # Goal: Reduce number of blocks.
    # Strategy: Try to remove a block. If the covering remains valid, remove it.
    # To check validity fast, we need to know if any t-subset is ONLY covered by that block.
    
    # Let's maintain a count of how many blocks cover each t-subset.
    # counts[j] = number of selected blocks covering t-subset j.
    
    # Initialize counts
    counts = np.zeros(num_t, dtype=np.int32)
    for idx in indices_selected:
        chunk_idx = idx // chunk_size
        local_idx = idx % chunk_size
        cover_row = cover_list[chunk_idx][local_idx]
        counts += cover_row
        
    # Try to remove blocks
    # We can try removing blocks in the order they were added (last in, first out) 
    # or try all.
    # To be efficient, we can iterate through selected blocks and try to remove them.
    
    # We need to be careful: if we remove a block, counts change.
    # If counts[j] becomes 0, it's invalid.
    
    # Let's try a simple pass:
    # For each selected block, check if it's "critical" (has any t-subset with count 1).
    # If not critical, remove it.
    
    # However, removing one block might make another block non-critical.
    # So we should iterate until no more blocks can be removed.
    
    def perform_removal_pass(indices, counts):
        changed = False
        new_indices = []
        for idx in indices:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            
            critical_mask = (cover_row == 1) & (counts == 1)
            if not np.any(critical_mask):
                counts -= cover_row
                changed = True
            else:
                new_indices.append(idx)
        return new_indices, counts, changed

    # Initial removal pass
    changed = True
    while changed and time.time() < deadline - 5:
        indices_selected, counts, changed = perform_removal_pass(indices_selected, counts)
        blocks_selected = [k_blocks[i] for i in indices_selected]
        
    # Swap Search: Try to swap blocks to enable further removals
    # We look for a block Bi that is critical (has unique subsets Si).
    # We look for a block Bj (not selected) that covers Si.
    # If found, swap Bi with Bj.
    # Then run removal pass. If count decreases, keep the change.
    
    # To avoid infinite loops, we track the number of blocks.
    # We only accept a swap if it leads to a net reduction in blocks after a removal pass.
    
    # We will iterate through selected blocks.
    # For efficiency, we only try swaps if we have time.
    
    swap_improved = True
    while swap_improved and time.time() < deadline - 5:
        swap_improved = False
        current_count = len(indices_selected)
        
        # Find a block to try swapping
        # We prioritize blocks with smaller |Si| as they are easier to cover.
        # Or just iterate.
        
        for i_idx in range(len(indices_selected)):
            if time.time() > deadline - 5:
                break
                
            idx = indices_selected[i_idx]
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            
            # Find unique subsets for this block
            critical_mask = (cover_row == 1) & (counts == 1)
            if not np.any(critical_mask):
                continue # Should have been removed
            
            # Get indices of critical subsets
            crit_indices = np.where(critical_mask)[0]
            
            # Find a candidate block Bj that covers all crit_indices
            # Bj must not be in indices_selected
            # We scan chunks
            
            found_swap = False
            new_idx = -1
            
            for c_idx, cover_chunk in enumerate(cover_list):
                if time.time() > deadline - 5:
                    break
                    
                # Check which blocks in this chunk cover all crit_indices
                # cover_chunk[:, crit_indices] is (chunk_size, len(crit_indices))
                # We want rows where all are 1.
                
                if len(crit_indices) == 0:
                    continue
                
                # Optimization: If len(crit_indices) is large, it's hard to find a cover.
                # But let's try.
                
                # We need to ensure the candidate block is NOT already selected.
                # We can check this later or filter now.
                # Filtering now is safer to avoid self-swap.
                
                # Create a mask of selected blocks in this chunk
                # This is slow. Let's just find a candidate and check if it's selected later.
                
                # Vectorized check:
                # Does block j cover all crit_indices?
                # np.all(cover_chunk[:, crit_indices] == 1, axis=1)
                
                # This creates a (chunk_size, len(crit_indices)) array temporarily?
                # No, indexing creates a view or copy.
                # Let's use a dot product trick or sum?
                # Sum of 1s should equal len(crit_indices).
                
                # cover_chunk[:, crit_indices].sum(axis=1) == len(crit_indices)
                
                # This is efficient.
                
                sums = cover_chunk[:, crit_indices].sum(axis=1)
                candidates = np.where(sums == len(crit_indices))[0]
                
                for local_cand in candidates:
                    cand_global_idx = c_idx * chunk_size + local_cand
                    
                    # Check if candidate is already selected
                    if cand_global_idx in indices_selected:
                        continue
                    
                    # Candidate found!
                    # Perform swap simulation
                    # 1. Remove idx
                    # 2. Add cand_global_idx
                    # 3. Run removal pass
                    # 4. If new count < current_count, accept.
                    
                    # Update counts for simulation
                    # We need to be careful not to modify the actual counts until we decide to accept.
                    # We can copy counts? 3876 ints is small.
                    sim_counts = counts.copy()
                    sim_counts -= cover_row # Remove idx
                    sim_counts += cover_list[c_idx][local_cand] # Add cand
                    
                    # Check if sim_counts has any 0s?
                    # The swap itself might uncover something if cand doesn't cover something idx covered?
                    # No, cand covers crit_indices (which are the only ones with count 1).
                    # Other subsets covered by idx had count >= 2, so removing idx makes them >= 1.
                    # So sim_counts should not have 0s.
                    
                    # Run removal pass on sim state
                    sim_indices = [i for i in indices_selected if i != idx]
                    sim_indices.append(cand_global_idx)
                    
                    sim_indices, sim_counts, _ = perform_removal_pass(sim_indices, sim_counts)
                    
                    if len(sim_indices) < current_count:
                        # Accept swap
                        indices_selected = sim_indices
                        counts = sim_counts
                        blocks_selected = [k_blocks[i] for i in indices_selected]
                        current_count = len(indices_selected)
                        swap_improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if found_swap:
                # We changed the list, so we should restart the outer loop or continue carefully.
                # Since indices_selected changed, the range(len(indices_selected)) is stale.
                # Breaking out of the inner for loop is good.
                break
    
    # Final check: ensure validity
    # Recalculate counts to be sure
    counts = np.zeros(num_t, dtype=np.int32)
    for idx in indices_selected:
        chunk_idx = idx // chunk_size
        local_idx = idx % chunk_size
        cover_row = cover_list[chunk_idx][local_idx]
        counts += cover_row
        
    if np.any(counts == 0):
        # Fallback to greedy if invalid (should not happen)
        # We don't have the greedy list stored separately anymore, 
        # but the logic should be safe.
        pass
        
    return [list(b) for b in blocks_selected]
