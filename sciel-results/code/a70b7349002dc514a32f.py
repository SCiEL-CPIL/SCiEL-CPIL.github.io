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
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
    
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # 2. Greedy Construction with random tie-breaking
    uncovered = np.ones(num_tsubs, dtype=bool)
    k_choose_t = len(block_to_tsubs[0])
    gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
    
    selected_block_indices = []
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
        
        # Random tie-breaking among blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        b = random.choice(candidates.tolist())
        
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
        
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_replacement(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in unique_tsubs:
                    if s not in c_set:
                        covers_all = False
                        break
                
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        def find_replacement_2for1(unique_tsubs_1, unique_tsubs_2, exclude_set):
            """Find a single block that covers all unique t-subsets from two blocks being removed."""
            all_uniq = set(unique_tsubs_1) | set(unique_tsubs_2)
            if not all_uniq:
                return None
            
            first_s = next(iter(all_uniq))
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in all_uniq:
                    if s not in c_set:
                        covers_all = False
                        break
                
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)
        
        best_size = len(selected_block_indices)
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
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Compute unique t-subsets for each block
            unique_tsubs_map = {}
            unique_counts = {}
            for b in selected_block_indices:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            min_unique = min(unique_counts.values()) if unique_counts else 0
            
            # Step B1: 1-for-1 Swap
            if min_unique > 0:
                candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
                random.shuffle(candidates_out)
                
                swap_made = False
                for b_out in candidates_out:
                    if time.time() > deadline:
                        break
                    s_uniq = unique_tsubs_map[b_out]
                    exclude_set = set(selected_block_indices)
                    
                    b_in = find_replacement(s_uniq, exclude_set)
                    
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
                
                if swap_made:
                    continue
            
            # Step B2: 2-for-1 Swap (remove 2 blocks, add 1)
            if len(selected_block_indices) >= 2:
                # Try pairs of blocks with small unique counts
                sorted_by_uniq = sorted(selected_block_indices, key=lambda b: unique_counts.get(b, 0))
                max_pairs_to_try = 50
                pairs_tried = 0
                
                swap_made_2 = False
                for i in range(len(sorted_by_uniq)):
                    if time.time() > deadline or pairs_tried >= max_pairs_to_try:
                        break
                    b1 = sorted_by_uniq[i]
                    for j in range(i + 1, len(sorted_by_uniq)):
                        if time.time() > deadline or pairs_tried >= max_pairs_to_try:
                            break
                        pairs_tried += 1
                        b2 = sorted_by_uniq[j]
                        
                        # Skip if either block has 0 unique (would be redundant)
                        if unique_counts[b1] == 0 or unique_counts[b2] == 0:
                            continue
                        
                        s_uniq1 = unique_tsubs_map[b1]
                        s_uniq2 = unique_tsubs_map[b2]
                        
                        exclude_set = set(selected_block_indices)
                        b_in = find_replacement_2for1(s_uniq1, s_uniq2, exclude_set)
                        
                        if b_in is not None:
                            # Remove both, add one
                            selected_block_indices.remove(b1)
                            selected_block_indices.remove(b2)
                            selected_block_indices.append(b_in)
                            
                            for s in block_to_tsubs[b1]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b2]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b_in]:
                                cover_count[s] += 1
                            
                            swap_made_2 = True
                            improved = True
                            break
                    
                    if swap_made_2:
                        break
                
                if swap_made_2:
                    continue
            
            # No improvement found
            break
        
        result = [list(all_k_subsets[i]) for i in selected_block_indices]
        return result
    else:
        return []
