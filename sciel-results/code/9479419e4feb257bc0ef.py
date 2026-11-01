import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (kill at 150s)
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    block_to_tsubs = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        
    # Precompute bitsets for t-subsets to blocks
    # num_blocks bits per t-subset
    num_words = (num_blocks + 63) // 64
    tsub_masks = [np.zeros(num_words, dtype=np.uint64) for _ in range(num_tsubs)]
    
    for b_idx, tsubs in enumerate(block_to_tsubs):
        word_idx = b_idx // 64
        bit_idx = b_idx % 64
        mask_bit = np.uint64(1) << np.uint64(bit_idx)
        for s_idx in tsubs:
            tsub_masks[s_idx][word_idx] |= mask_bit
            
    # Also keep list of blocks for each t-subset for quick access if needed, 
    # but bitsets are better for intersection.
    # We also need to map block index to t-subsets for updates.
    
    # 2. Greedy Construction
    uncovered = np.ones(num_tsubs, dtype=bool)
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        b = np.argmax(gains)
        
        if gains[b] == 0:
            break
        
        selected_block_indices.append(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                # Update gains for blocks covering this t-subset
                # We need list of blocks covering s_idx for fast iteration
                # We can derive this from tsub_masks, but iterating bits is slow in Python.
                # Let's keep the list version for greedy updates as it was fast enough.
                pass 
        
    # Re-implement greedy with lists for speed, as bitset iteration in Python is slow.
    # The previous code had tsub_to_blocks_np. Let's restore that.
    
    # Redo precomputation for lists
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Reset Greedy
    uncovered = np.ones(num_tsubs, dtype=bool)
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        b = np.argmax(gains)
        
        if gains[b] == 0:
            break
        
        selected_block_indices.append(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                gains[tsub_to_blocks_np[s_idx]] -= 1
                
    if not uncovered.any():
        # 3. Local Optimization
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        selected_set = set(selected_block_indices)
        
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
            
        def find_replacement_1for1(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            # Use bitsets for intersection
            mask = np.full(num_words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
            for s in unique_tsubs:
                mask &= tsub_masks[s]
            
            # Get candidates
            candidates = np.flatnonzero(mask)
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                valid_candidates.append(c)
                
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        def find_replacement_2for1(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            mask = np.full(num_words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
            for s in unique_tsubs:
                mask &= tsub_masks[s]
            
            candidates = np.flatnonzero(mask)
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                valid_candidates.append(c)
                
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step B: 1-for-1 Swap
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_block_indices:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            min_unique = min(unique_counts.values())
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                if not s_uniq:
                    continue # Should be redundant, handled above
                
                b_in = find_replacement_1for1(s_uniq, selected_set)
                
                if b_in is not None:
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    selected_set.remove(b_out)
                    selected_set.add(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break
            
            if swap_made:
                continue
                
            # Step C: 2-for-1 Swap
            # Try to find two blocks b1, b2 such that their unique t-subsets
            # can be covered by a single new block.
            
            # Heuristic: Pick pairs with small unique counts
            # To keep it fast, we can limit the search or just try random pairs?
            # Given N~150, N^2/2 ~ 11k pairs.
            # For each pair, check intersection.
            
            # We need to be careful with time.
            # Let's try a limited number of random pairs or systematic.
            
            # Calculate unique t-subsets for all blocks again (or use cache from B)
            # unique_tsubs_map is up to date.
            
            # Sort blocks by unique count to try "easy" pairs first?
            # Or just iterate.
            
            found_2for1 = False
            # To avoid O(N^2) being too slow if it fails often, we can add a time check inside.
            
            # Let's try systematic pairs but break early if time is tight.
            n_sel = len(selected_block_indices)
            
            for i in range(n_sel):
                if time.time() > deadline - 5: break
                b1 = selected_block_indices[i]
                u1 = unique_tsubs_map[b1]
                
                for j in range(i+1, n_sel):
                    if time.time() > deadline - 5: break
                    b2 = selected_block_indices[j]
                    u2 = unique_tsubs_map[b2]
                    
                    # Union of unique t-subsets
                    # Note: if a t-subset is unique to b1, it is not covered by b2 (since b2 is selected, if it covered it, cover_count would be >=2? 
                    # No, cover_count[s] is total. If s is in b1 and b2, cover_count[s] >= 2. 
                    # So u1 and u2 are disjoint sets of t-subsets?
                    # Yes, if s is in u1, cover_count[s]==1 and s in b1. Since b2 is selected, if s in b2, cover_count[s] would be at least 2 (from b1 and b2). 
                    # So u1 and u2 are disjoint.
                    
                    u_union = u1 + u2
                    
                    if not u_union:
                        # Both redundant? Handled in A.
                        continue
                    
                    b_in = find_replacement_2for1(u_union, selected_set)
                    
                    if b_in is not None:
                        # Perform swap
                        selected_block_indices.remove(b1)
                        selected_block_indices.remove(b2)
                        selected_block_indices.append(b_in)
                        
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        selected_set.add(b_in)
                        
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b_in]:
                            cover_count[s] += 1
                        
                        found_2for1 = True
                        improved = True
                        break
                
                if found_2for1:
                    break
            
            if not found_2for1:
                # No 2-for-1 found. Stop?
                # We could try 1-for-2? No, increases size.
                # We could try 3-for-1? Too complex.
                break
        
        result = [list(all_k_subsets[i]) for i in selected_block_indices]
        return result
    else:
        return []
