import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage
    block_to_tsubs = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
    
    # Inverse mapping
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
    
    # Bitmask representation: for each block, a bitmask of which t-subsets it covers
    # Since num_tsubs can be large, we use multiple uint64 words
    num_words = (num_tsubs + 63) // 64
    block_bitmasks = np.zeros((num_blocks, num_words), dtype=np.uint64)
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            word = s_idx >> 6
            bit = s_idx & 63
            block_bitmasks[b_idx, word] |= (1 << bit)
    
    # For each t-subset, list of blocks that cover it
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    def greedy_construction(randomize_ties=False, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            if randomize_ties:
                if random.random() < 0.3:
                    top_k = 15
                    if num_blocks > top_k:
                        top_indices = np.argpartition(gains, -top_k)[-top_k:]
                        b = random.choice(top_indices)
                    else:
                        b = np.argmax(gains)
                else:
                    b = np.argmax(gains)
            else:
                b = np.argmax(gains)
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None
            
        return selected_block_indices

    def local_optimize(blocks, max_time=10.0):
        if not blocks:
            return []
        
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        local_deadline = time.time() + max_time
        
        improved = True
        while improved and time.time() < local_deadline:
            improved = False
            
            # Step A: Redundancy Removal
            redundant_blocks = []
            for b in selected_list:
                is_red = True
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        is_red = False
                        break
                if is_red:
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step B: 1-swap
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_list:
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_list.append(s)
                unique_tsubs_map[b] = u_list
                unique_counts[b] = len(u_list)
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs_map[b_out]
                if not s_uniq:
                    continue
                
                # Use bitmask to quickly check candidates
                first_s = s_uniq[0]
                candidates = tsub_to_blocks_np[first_s]
                
                # Build bitmask of required t-subsets
                req_mask_words = num_words
                req_mask = np.zeros(num_words, dtype=np.uint64)
                for s in s_uniq:
                    word = s >> 6
                    bit = s & 63
                    req_mask[word] |= (1 << bit)
                
                found = False
                for c in candidates:
                    if c in selected_set:
                        continue
                    # Check if block c covers all required t-subsets using bitmask
                    covers_all = True
                    for w in range(num_words):
                        if (block_bitmasks[c, w] & req_mask[w]) != req_mask[w]:
                            covers_all = False
                            break
                    if covers_all:
                        # Perform swap
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        selected_list.append(c)
                        selected_set.add(c)
                        
                        for s in block_to_tsubs[b_out]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[c]:
                            cover_count[s] += 1
                        
                        swap_made = True
                        found = True
                        break
                
                if found:
                    improved = True
                    break
            
            if swap_made:
                continue
            
            # Step C: 2-swap (remove 2 blocks, add 1 block that covers all their unique t-subsets)
            # Only try if we have time
            if time.time() < local_deadline - 0.5 and len(selected_list) >= 3:
                two_swap_made = False
                # Try pairs of blocks with few unique t-subsets
                sorted_by_unique = sorted(selected_list, key=lambda b: unique_counts.get(b, 0))
                num_tried = 0
                max_pairs = 50
                
                for i in range(min(len(sorted_by_unique), 10)):
                    if two_swap_made or num_tried >= max_pairs:
                        break
                    b1 = sorted_by_unique[i]
                    u1 = unique_tsubs_map.get(b1, [])
                    for j in range(i + 1, min(len(sorted_by_unique), 15)):
                        if two_swap_made or num_tried >= max_pairs:
                            break
                        num_tried += 1
                        b2 = sorted_by_unique[j]
                        u2 = unique_tsubs_map.get(b2, [])
                        
                        # Combined unique t-subsets
                        combined = list(set(u1) | set(u2))
                        if not combined:
                            continue
                        
                        # Find a block that covers all combined unique t-subsets
                        first_s = combined[0]
                        candidates = tsub_to_blocks_np[first_s]
                        
                        req_mask = np.zeros(num_words, dtype=np.uint64)
                        for s in combined:
                            word = s >> 6
                            bit = s & 63
                            req_mask[word] |= (1 << bit)
                        
                        for c in candidates:
                            if c in selected_set:
                                continue
                            covers_all = True
                            for w in range(num_words):
                                if (block_bitmasks[c, w] & req_mask[w]) != req_mask[w]:
                                    covers_all = False
                                    break
                            if covers_all:
                                # Perform 2-swap
                                selected_list.remove(b1)
                                selected_set.remove(b1)
                                selected_list.remove(b2)
                                selected_set.remove(b2)
                                selected_list.append(c)
                                selected_set.add(c)
                                
                                for s in block_to_tsubs[b1]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[b2]:
                                    cover_count[s] -= 1
                                for s in block_to_tsubs[c]:
                                    cover_count[s] += 1
                                
                                two_swap_made = True
                                improved = True
                                break
                        
                        if two_swap_made:
                            break
                
                if two_swap_made:
                    continue
            
            break
        
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start with adaptive time allocation
    # First run: longer local search
    blocks = greedy_construction(randomize_ties=False, random_seed=42)
    if blocks is not None:
        optimized_blocks = local_optimize(blocks, max_time=15.0)
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
    
    # Subsequent runs with shorter local search to fit more restarts
    for i in range(1, 30):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate more time to local search as we get closer to deadline
        if i <= 5:
            ls_time = min(8.0, remaining * 0.4)
        elif i <= 15:
            ls_time = min(5.0, remaining * 0.3)
        else:
            ls_time = min(3.0, remaining * 0.2)
        
        if ls_time < 1.0:
            break
            
        blocks = greedy_construction(randomize_ties=True, random_seed=42 + i)
        if blocks is None:
            continue
            
        optimized_blocks = local_optimize(blocks, max_time=ls_time)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
        
        # Early stopping if we've found a very good solution
        if best_len <= 145:
            break
    
    if best_solution is None:
        return []
        
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
