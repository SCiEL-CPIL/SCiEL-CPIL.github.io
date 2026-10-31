# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: 150s wall clock limit, leave 5s safety margin
    deadline = time.time() + 145.0
    
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
    # We will store these in a flat array with offsets for efficient numpy slicing
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            indices.append(idx_of[m])
        k_sub_tsub_indices.append(indices)
    
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # State
    uncovered = np.ones(n_tsub, dtype=bool)
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    blocks_selected = []
    selected_set = set()
    
    # 2. Vectorized Batched Greedy
    # Instead of calculating gains for all n_ksub blocks in a Python loop (slow),
    # we process them in batches using numpy operations.
    
    batch_size = 5000 # Tuned to fit in cache and memory efficiently
    
    while uncovered.any() and time.time() < deadline - 10.0:
        # We need to find the block with the maximum gain.
        # Since we can't vectorize the "argmax" over all blocks easily without a huge matrix,
        # we process in batches and keep the global best.
        
        best_gain = -1
        best_idx = -1
        best_candidates = []
        
        for start_batch in range(0, n_ksub, batch_size):
            end_batch = min(start_batch + batch_size, n_ksub)
            
            # Get indices for this batch of blocks
            # We need the t-subset indices for blocks in [start_batch, end_batch)
            # offsets_arr gives us the start/end in all_indices_arr for each block.
            
            # The start of the first block in batch
            batch_start_idx = offsets_arr[start_batch]
            # The end of the last block in batch
            batch_end_idx = offsets_arr[end_batch]
            
            # We need to calculate the sum of uncovered t-subsets for each block in the batch.
            # This is tricky with the flat array because block sizes are constant (C(k,t)), 
            # but let's assume constant size for simplicity in vectorization.
            # Actually, C(k,t) is constant. Let's call it ckt.
            
            ckt = offsets_arr[1] - offsets_arr[0]
            
            if end_batch > start_batch:
                # Extract the submatrix of t-subset indices for this batch
                # Shape: (batch_count, ckt)
                # We can do this by reshaping the slice of all_indices_arr
                batch_indices = all_indices_arr[batch_start_idx:batch_end_idx]
                
                if len(batch_indices) == 0:
                    continue
                    
                # Reshape to (batch_count, ckt)
                # This works because all blocks have exactly ckt t-subsets
                batch_matrix = batch_indices.reshape(-1, ckt)
                
                # Calculate gains: sum of uncovered[tsub_idx] for each block
                # uncovered is a 1D array. We index it with batch_matrix.
                # This creates a (batch_count, ckt) boolean array.
                # Summing along axis 1 gives the gain for each block.
                
                # Optimization: Use np.take or direct indexing
                # uncovered[batch_matrix] is the fastest way in numpy
                batch_gains = np.sum(uncovered[batch_matrix], axis=1)
                
                # Find max gain in this batch
                local_max = np.max(batch_gains)
                
                if local_max > best_gain:
                    best_gain = local_max
                    # Find indices in the batch that have this max gain
                    local_candidates = np.where(batch_gains == best_gain)[0]
                    # Map back to global indices
                    best_candidates = [start_batch + i for i in local_candidates]
                elif local_max == best_gain:
                    local_candidates = np.where(batch_gains == best_gain)[0]
                    best_candidates.extend([start_batch + i for i in local_candidates])
        
        if best_gain <= 0:
            break
            
        # Pick a candidate
        # To improve quality, we can use a tie-breaker or random selection.
        # Random selection helps escape symmetries.
        if best_candidates:
            # Occasionally pick the first (deterministic) to ensure progress, 
            # but mostly random to explore.
            if random.random() < 0.1:
                best_idx = best_candidates[0]
            else:
                best_idx = random.choice(best_candidates)
        else:
            break
        
        # Add block
        blocks_selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update state
        start = offsets_arr[best_idx]
        end = offsets_arr[best_idx+1]
        idxs = all_indices_arr[start:end]
        
        # Update cover_count
        cover_count[idxs] += 1
        
        # Update uncovered
        uncovered[idxs] = False
        
    # 3. Repair Phase
    # If there are still uncovered t-subsets (shouldn't happen if greedy finished, but just in case)
    uncovered_indices = np.where(uncovered)[0]
    
    # Build mask_to_idx for fast lookup if needed, but we have all_k_subsets
    # We can just iterate through all_k_subsets to find a covering block for the remaining t-subsets
    # However, since greedy is very strong, this list should be empty or very small.
    
    for idx in uncovered_indices:
        if time.time() > deadline - 5.0:
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
        # This is a linear search, but only for very few items
        mask = 0
        for p in block:
            mask |= (1 << p)
        
        # We need a fast lookup. Let's build it once if we haven't.
        # To save time in the loop, we'll do a linear search here assuming it's rare.
        # Or better, precompute a dict.
        
        # Precomputing a dict for 50k items is fast (<0.1s)
        # Let's do it lazily if the repair list is not empty.
        
    if uncovered_indices:
         # Build lookup
         mask_to_idx = {}
         for i, comb in enumerate(all_k_subsets):
             m = 0
             for p in comb:
                 m |= (1 << p)
             mask_to_idx[m] = i
             
         for idx in uncovered_indices:
             if time.time() > deadline - 5.0:
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
                 
                 start = offsets_arr[b_idx]
                 end = offsets_arr[b_idx+1]
                 idxs = all_indices_arr[start:end]
                 cover_count[idxs] += 1
                 uncovered[idxs] = False

    # 4. Local Search / Swap Phase
    # Try to replace pairs of blocks with a single block or reduce the set.
    # Strategy: 
    # 1. Remove redundant blocks (blocks that cover no unique t-subsets).
    # 2. Try 2-for-1 swaps: If two blocks B1, B2 cover a set of t-subsets S1, S2.
    #    If there exists a block B3 such that S1 U S2 is covered by B3 plus the rest of the solution,
    #    and B3 is not in the solution, we can replace B1, B2 with B3.
    #    This is complex. A simpler heuristic:
    #    Iterate through blocks. For each block B, check if removing it leaves some t-subsets uncovered.
    #    If so, try to find a single block B' that covers all those uncovered t-subsets.
    #    If found, replace B with B'.
    
    # Let's start with redundancy removal, which is safe and fast.
    
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        # Iterate in reverse to make removal easier
        for i in range(len(blocks_selected) - 1, -1, -1):
            b_idx = blocks_selected[i]
            
            # Check if redundant
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            idxs = all_indices_arr[start:end]
            
            # If any t-subset in this block has count == 1, it's not redundant
            if np.any(cover_count[idxs] == 1):
                continue
            
            # It is redundant
            cover_count[idxs] -= 1
            blocks_selected.pop(i)
            selected_set.remove(b_idx)
            changed = True
            # Don't break, continue checking other blocks in this pass
            # But we need to be careful with indices. Popping from end is safe.
            
    # 5. 2-for-1 Swap Attempt (Simplified)
    # We try to find if two blocks can be replaced by one.
    # This is expensive. We will only do this if we have time.
    
    if time.time() < deadline - 10.0:
        # Create a mapping from t-subset index to list of blocks covering it?
        # No, that's too big.
        
        # Instead, we do a random sampling of pairs.
        # Pick two random blocks B1, B2.
        # Calculate the set of t-subsets they cover: S12.
        # If |S12| <= C(k,t), it might be coverable by one block.
        # Actually, any 2 blocks cover at most 2*C(k,t) t-subsets.
        # A single block covers C(k,t).
        # So we can only replace if S12 is a subset of the t-subsets of some block B3.
        # This implies S12 must be small.
        
        # A better heuristic:
        # For each block B, identify the "unique" t-subsets it covers (count == 1).
        # If a block has no unique t-subsets, it was already removed.
        # So every remaining block has at least one unique t-subset.
        # To merge B1 and B2, we need a B3 that covers Unique(B1) U Unique(B2) U (Shared(B1,B2) that might become unique if B1,B2 removed? No, Shared are covered by others too).
        
        # Actually, if we remove B1 and B2, the t-subsets that become uncovered are exactly those with count 1 in B1 or B2.
        # Let U1 = {tsub | count[tsub]==1 and tsub in B1}
        # Let U2 = {tsub | count[tsub]==1 and tsub in B2}
        # We need a block B3 such that U1 U U2 is a subset of B3.
        
        # Let's try a few random pairs.
        
        for _ in range(100): # Limit attempts
            if time.time() > deadline - 2.0:
                break
            if len(blocks_selected) < 2:
                break
                
            i, j = random.sample(range(len(blocks_selected)), 2)
            b1 = blocks_selected[i]
            b2 = blocks_selected[j]
            
            # Get unique t-subsets for b1 and b2
            start1 = offsets_arr[b1]
            end1 = offsets_arr[b1+1]
            idxs1 = all_indices_arr[start1:end1]
            
            start2 = offsets_arr[b2]
            end2 = offsets_arr[b2+1]
            idxs2 = all_indices_arr[start2:end2]
            
            # Find indices where count is 1
            unique1 = idxs1[cover_count[idxs1] == 1]
            unique2 = idxs2[cover_count[idxs2] == 1]
            
            # Combine unique sets
            # We need to check if there is a block B3 that covers all of unique1 and unique2.
            # This is a set cover check.
            # Let target = set(unique1) U set(unique2)
            
            # Optimization: If |target| > C(k,t), impossible.
            # But even if small, checking all B3 is slow.
            
            # Heuristic: Pick one random t-subset from target. Find all blocks covering it.
            # Check if any of those blocks cover the rest of target.
            
            if len(unique1) + len(unique2) == 0:
                continue
                
            target_list = list(set(unique1.tolist()) | set(unique2.tolist()))
            if not target_list:
                continue
            
            # Pick a pivot t-subset
            pivot = random.choice(target_list)
            
            # Find all blocks covering pivot
            # We don't have an inverted index. Building one is expensive.
            # Let's skip this complex swap for now and rely on greedy + redundancy.
            # The "Local search with redundancy removal and 1-for-1 swaps" finding suggests
            # that swaps are useful. But 2-for-1 is hard without inverted index.
            
            # Let's try 1-for-1 swap:
            # Replace B1 with B3 if B3 covers all unique t-subsets of B1.
            # This doesn't reduce block count, but might enable further redundancy removal.
            
            unique1_list = unique1.tolist()
            if not unique1_list:
                continue
                
            pivot = random.choice(unique1_list)
            
            # We need to find blocks covering pivot.
            # Let's just iterate through all blocks? No, too slow.
            
            # Given the time constraints and complexity, we will stick to the Greedy + Redundancy Removal.
            # The vectorization in Greedy is the main structural improvement over the previous scalar loop.
            pass

    # 6. Final Conversion
    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
