# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: leave 10 seconds for final repair and return
    deadline = time.time() + 110.0
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks and indices
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    
    n_tsub = len(tsub_masks)
    idx_of = {m: i for i, m in enumerate(tsub_masks)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Precompute t-subset indices for each k-subset
    # k_sub_tsub_indices[i] is a list of t-subset indices covered by block i
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            indices.append(idx_of[m])
        k_sub_tsub_indices.append(indices)
    
    # Convert to numpy arrays for faster access if needed, but lists are fine for iteration
    # We will use the list of lists.
    
    # State
    uncovered = np.ones(n_tsub, dtype=bool)
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    blocks_selected = []
    
    # 2. Greedy Loop
    # Optimization: Pre-compute the length of indices (constant C(k,t))
    # We will calculate gain for all blocks in a loop.
    
    # To speed up, we can store the indices as a flat array with offsets
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Greedy Selection
    while uncovered.any() and time.time() < deadline - 5.0:
        gains = np.zeros(n_ksub, dtype=np.int32)
        
        # Calculate gains
        # This loop is the bottleneck.
        # Using local variables for speed
        unc = uncovered
        ind_arr = all_indices_arr
        off_arr = offsets_arr
        
        for i in range(n_ksub):
            start = off_arr[i]
            end = off_arr[i+1]
            if start < end:
                gains[i] = np.sum(unc[ind_arr[start:end]])
        
        max_gain = np.max(gains)
        
        if max_gain == 0:
            break
        
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Pick one. To improve quality, we can use a tie-breaker:
        # Prefer blocks that cover t-subsets with fewer "remaining" covering options?
        # For simplicity and speed, we pick the first one (deterministic) or random.
        # Let's pick the one with the smallest index to be deterministic.
        best_idx = candidates[0]
        
        # Add block
        blocks_selected.append(best_idx)
        
        # Update state
        start = off_arr[best_idx]
        end = off_arr[best_idx+1]
        idxs = ind_arr[start:end]
        
        # Update cover_count
        cover_count[idxs] += 1
        
        # Update uncovered
        # Only set to False if it was True?
        # np.where(uncovered[idxs], False, uncovered[idxs])?
        # Simply:
        uncovered[idxs] = False
        
    # 3. Repair Phase (if deadline hit or uncovered remains)
    uncovered_indices = np.where(uncovered)[0]
    
    for idx in uncovered_indices:
        if time.time() > deadline - 2.0:
            break
            
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        
        if need > 0:
            # Pick random extra points
            extra = random.sample(remaining_points, need)
        else:
            extra = []
        
        block = have + extra
        block.sort()
        
        # Find index of this block in all_k_subsets
        # We can use a dictionary for mask to index
        mask = 0
        for p in block:
            mask |= (1 << p)
        
        # We need a fast lookup.
        # Let's build mask_to_idx once outside?
        # It's 50k entries.
        
        # For now, linear search is too slow if many repairs.
        # But repairs should be few if greedy works.
        
        # Let's precompute mask_to_idx
        # We'll do this in the main setup.
        
    # 4. Reduction Phase
    # Remove redundant blocks
    # This is safe and fast.
    
    # We need mask_to_idx for the repair phase. Let's build it now if not done.
    # Actually, let's just build it at the start.
    
    # Re-do precomputation with mask_to_idx
    mask_to_idx = {}
    for i, comb in enumerate(all_k_subsets):
        m = 0
        for p in comb:
            m |= (1 << p)
        mask_to_idx[m] = i

    # Repair again with proper lookup
    uncovered_indices = np.where(uncovered)[0]
    for idx in uncovered_indices:
        if time.time() > deadline - 2.0:
            break
            
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        
        if need > 0:
            extra = random.sample(remaining_points, need)
        else:
            extra = []
        
        block = have + extra
        block.sort()
        
        mask = 0
        for p in block:
            mask |= (1 << p)
        
        b_idx = mask_to_idx.get(mask)
        if b_idx is not None:
            # Check if already selected
            # To check if already selected, we can maintain a set of selected indices
            # Let's maintain a set
            pass

    # Let's refactor to maintain a set of selected indices for O(1) check
    
    # Re-run the logic with a set
    selected_set = set(blocks_selected)
    
    # The repair loop above didn't add to blocks_selected properly.
    # Let's do a clean repair.
    
    # Clear the partial repair if any (it was just a sketch)
    # Actually, the loop above didn't modify blocks_selected.
    
    # Let's perform the repair correctly.
    uncovered_indices = np.where(uncovered)[0]
    for idx in uncovered_indices:
        if time.time() > deadline - 2.0:
            break
            
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        
        if need > 0:
            extra = random.sample(remaining_points, need)
        else:
            extra = []
        
        block = have + extra
        block.sort()
        
        mask = 0
        for p in block:
            mask |= (1 << p)
        
        b_idx = mask_to_idx.get(mask)
        if b_idx is not None and b_idx not in selected_set:
            blocks_selected.append(b_idx)
            selected_set.add(b_idx)
            
            # Update cover_count and uncovered
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            idxs = all_indices_arr[start:end]
            cover_count[idxs] += 1
            uncovered[idxs] = False

    # 5. Reduction Phase
    # Remove redundant blocks
    # We iterate until no more changes or time limit
    
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        # Iterate in reverse to make removal easier
        for i in range(len(blocks_selected) - 1, -1, -1):
            b_idx = blocks_selected[i]
            
            # Check if redundant
            redundant = True
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            idxs = all_indices_arr[start:end]
            
            # Check if any t-subset in this block has count == 1
            # If yes, it's not redundant
            # np.any(cover_count[idxs] == 1)
            if np.any(cover_count[idxs] == 1):
                redundant = False
            
            if redundant:
                # Remove block
                # Decrement counts
                cover_count[idxs] -= 1
                blocks_selected.pop(i)
                selected_set.remove(b_idx)
                changed = True
                break # Restart loop

    # 6. Final Conversion
    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
