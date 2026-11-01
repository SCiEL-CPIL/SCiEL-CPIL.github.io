import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers as a bitmask
    # For v=19, k=7, t=4: C(19,4) = 3876 t-subsets, which fits in a 4096-bit integer
    block_bitmasks = [0] * num_blocks
    block_to_tsubs = []
    for i, block in enumerate(k_subsets):
        mask = 0
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            mask |= (1 << idx)
            covered.append(idx)
        block_bitmasks[i] = mask
        block_to_tsubs.append(covered)
        
    # Precompute for each t-subset, which blocks cover it (for local search)
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert tsub_to_blocks lists to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Track uncovered t-subsets as a bitmask
    full_mask = (1 << num_tsubs) - 1
    uncovered_mask = full_mask
    
    selected_block_indices = []
    
    # 1. Greedy Construction using bitmask popcount
    while uncovered_mask != 0:
        if time.time() > deadline - 20:  # Leave time for local search
            break
            
        # Find the block with the maximum gain (popcount of block_mask & uncovered_mask)
        best_gain = 0
        best_b = -1
        for b in range(num_blocks):
            gain = (block_bitmasks[b] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
            if gain == 0:
                continue
                
        if best_b == -1 or best_gain == 0:
            break
            
        selected_block_indices.append(best_b)
        uncovered_mask &= ~block_bitmasks[best_b]
        
    # If we didn't finish greedy (shouldn't happen for v=19), force completion
    while uncovered_mask != 0:
        best_gain = 0
        best_b = -1
        for b in range(num_blocks):
            gain = (block_bitmasks[b] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
        if best_b == -1:
            break
        selected_block_indices.append(best_b)
        uncovered_mask &= ~block_bitmasks[best_b]
        
    # 2. Local Optimization
    if time.time() < deadline - 5:
        # Maintain cover_count for each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Helper to check if a block is redundant
        def is_redundant(b_idx, cover_count):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        # Helper to find a block covering all t-subsets in a list
        def find_covering_block(tsubs_to_cover, exclude_set):
            if not tsubs_to_cover:
                return None
            
            first_s = tsubs_to_cover[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                covers_all = True
                for s in tsubs_to_cover[1:]:
                    if c not in tsub_to_blocks[s]:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Step 2a: Redundancy Removal
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b, cover_count):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step 2b: Try Swaps
            unique_counts = {}
            unique_tsubs = {}
            
            for b in selected_block_indices:
                u_count = 0
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_count += 1
                        u_list.append(s)
                unique_counts[b] = u_count
                unique_tsubs[b] = u_list
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs[b_out]
                
                if not s_uniq:
                    # This block is fully redundant, remove it
                    selected_block_indices.remove(b_out)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    swap_made = True
                    improved = True
                    break
                    
                exclude_set = set(selected_block_indices)
                b_in = find_covering_block(s_uniq, exclude_set)
                
                if b_in is not None:
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break
            
            if not swap_made:
                break

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    # Deduplicate just in case
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(blk)
        if key not in seen:
            seen.add(key)
            dedup.append(blk)
            
    return dedup
