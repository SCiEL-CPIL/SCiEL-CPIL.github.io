import itertools
import sys
import random
import time

def construct(v, k, t):
    points = range(v)
    
    # 1. Enumerate all t-subsets and map them to indices
    t_subs = list(itertools.combinations(points, t))
    n_t_subs = len(t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Enumerate all k-blocks and compute their coverage masks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute masks for each k-block
    block_masks = [0] * n_k_blocks
    # Also store the list of t-subset indices covered by each block for faster iteration
    block_t_indices = [[] for _ in range(n_k_blocks)]
    
    # Inverse index: for each t-subset, which blocks cover it?
    # t_sub_blocks[i] = list of block indices j such that t_subs[i] is subset of k_blocks[j]
    t_sub_blocks = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(k_blocks):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
            t_sub_blocks[idx].append(i)
        block_masks[i] = mask
        block_t_indices[i] = indices

    # 3. Greedy Set Cover for Initialization
    uncovered_mask = (1 << n_t_subs) - 1
    selected_indices = []
    
    # Optimization for Greedy: 
    # Instead of scanning all blocks every time, we can just scan all. 
    # 75k blocks * 335 steps is ~25M ops. 
    # Bitwise AND + popcount on Python ints is fast.
    
    while uncovered_mask:
        best_idx = -1
        max_gain = 0
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > max_gain:
                max_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        selected_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # Convert to set for O(1) lookup
    current_set = set(selected_indices)
    
    # Maintain counts: count[i] = number of blocks in current_set covering t_sub i
    counts = [0] * n_t_subs
    for idx in current_set:
        for t_idx in block_t_indices[idx]:
            counts[t_idx] += 1
            
    # 4. Local Search: Reduce and Swap
    # Goal: Minimize len(current_set)
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        # Phase 1: Try to remove any block
        removed_any = False
        # Randomize order to break symmetry? Or just iterate.
        # Iterate over a copy because we modify the set
        for idx in list(current_set):
            # Check if idx is removable
            # Removable if for all t_idx in block_t_indices[idx], counts[t_idx] > 1
            removable = True
            for t_idx in block_t_indices[idx]:
                if counts[t_idx] <= 1:
                    removable = False
                    break
            if removable:
                # Remove idx
                current_set.remove(idx)
                for t_idx in block_t_indices[idx]:
                    counts[t_idx] -= 1
                removed_any = True
                # If we removed one, we can restart the scan or continue.
                # Continuing is fine, but the set changed.
                # Since we are iterating over a list copy, it's safe.
                
        if not removed_any:
            # Phase 2: Local Minimum. Try to swap to enable removal.
            # Pick a random block to remove (B_out)
            # We want to swap B_out with B_in such that validity is maintained.
            # Validity condition: Any t-subset uniquely covered by B_out must be covered by B_in.
            
            # To make this efficient, we can pick a random B_out, find its unique t-subsets,
            # and then find a B_in that covers them.
            
            # If we just pick random, it might take long.
            # Better: Pick a B_out that has *some* unique t-subsets but not too many?
            # Or just pick random.
            
            # Let's try a few random swaps per local minimum encounter
            swap_done = False
            attempts = 0
            max_attempts = 100
            
            while not swap_done and attempts < max_attempts:
                attempts += 1
                # Pick random B_out from current_set
                # Use random.choice on a list? current_set is a set.
                # Convert to list? Slow.
                # We can maintain a list of current indices.
                # Let's maintain current_list parallel to current_set?
                # Or just convert current_set to list once per phase 2 entry.
                # Since Phase 1 removes items, current_set changes.
                # Let's just use random.sample(list(current_set), 1)
                
                b_out_list = list(current_set)
                if not b_out_list: break
                b_out = random.choice(b_out_list)
                
                # Find unique t-subsets for b_out
                unique_t = []
                for t_idx in block_t_indices[b_out]:
                    if counts[t_idx] == 1:
                        unique_t.append(t_idx)
                        
                if not unique_t:
                    # This block is removable, but Phase 1 missed it?
                    # Should not happen if Phase 1 was thorough.
                    # But if counts changed? No, counts are consistent.
                    # If unique_t is empty, it means all t-subs are covered by >=1 other block.
                    # So it should have been removed in Phase 1.
                    # Maybe we just exited Phase 1 without checking this one?
                    # No, Phase 1 iterates all.
                    # So this case implies an error or race condition?
                    # Actually, if we removed other blocks, counts changed.
                    # But we are in Phase 2 because Phase 1 removed NOTHING.
                    # So counts are stable.
                    # So if unique_t is empty, it is removable.
                    # Why didn't Phase 1 remove it?
                    # Maybe it did, and we are iterating over a stale list?
                    # No, we created list(current_set) at start of Phase 1.
                    # If we removed items, the list is stale but we only check indices in the list.
                    # If an index was removed, it's not in current_set, so we shouldn't pick it.
                    # random.choice(b_out_list) might pick a removed one if b_out_list is stale?
                    # Yes, b_out_list is created at the start of the while loop? No, inside the while loop.
                    # So it's fresh.
                    # If unique_t is empty, remove it immediately.
                    current_set.remove(b_out)
                    for t_idx in block_t_indices[b_out]:
                        counts[t_idx] -= 1
                    removed_any = True # Force restart of outer loop
                    swap_done = True
                    break
                
                # Find B_in that covers all unique_t
                # Intersection of t_sub_blocks[t_idx] for t_idx in unique_t
                # Start with the block list of the first unique t-subset
                candidates = t_sub_blocks[unique_t[0]]
                
                # Filter candidates:
                # 1. Must not be in current_set
                # 2. Must cover all other unique_t
                # 3. Ideally, we want a B_in that doesn't introduce new problems?
                #    Actually, any B_in that covers unique_t keeps the cover valid.
                
                found_b_in = -1
                for b_in in candidates:
                    if b_in in current_set:
                        continue
                    # Check if b_in covers all unique_t
                    valid = True
                    for t_idx in unique_t:
                        # We know b_in covers unique_t[0] by construction.
                        # Check others.
                        # We can check if t_idx is in block_t_indices[b_in]?
                        # That's O(56).
                        # Or use mask: (block_masks[b_in] >> t_idx) & 1
                        if not (block_masks[b_in] >> t_idx) & 1:
                            valid = False
                            break
                    if valid:
                        found_b_in = b_in
                        break
                        
                if found_b_in != -1:
                    # Perform Swap
                    current_set.remove(b_out)
                    for t_idx in block_t_indices[b_out]:
                        counts[t_idx] -= 1
                        
                    current_set.add(found_b_in)
                    for t_idx in block_t_indices[found_b_in]:
                        counts[t_idx] += 1
                        
                    swap_done = True
                    # After swap, we might have new removals possible.
                    # So break out of swap loop and go back to Phase 1.
                    break
            if not swap_done and not removed_any:
                # Failed to find a swap. 
                # We are stuck in a deep local minimum.
                # We could try a "kick": add a random block, then reduce?
                # Adding a block increases size. We need to remove 2 to get back to -1.
                # This is risky.
                # Given time constraints, we might just stop or try harder swaps.
                # Let's try a few more random B_out choices if we failed.
                # The loop above already tried 100 random B_outs.
                # If that fails, we are likely at a global minimum or very close.
                # We can break to avoid wasting time.
                break
                
        if removed_any:
            # Restart the while loop to try more removals
            continue
        else:
            # If we didn't remove and didn't swap, we break?
            # The code above breaks if swap_done is False and removed_any is False.
            # If swap_done is True, we continue to Phase 1.
            # If we break, we are done.
            break

    return [list(k_blocks[i]) for i in current_set]
