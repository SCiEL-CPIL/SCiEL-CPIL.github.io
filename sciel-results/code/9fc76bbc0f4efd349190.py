import itertools
import numpy as np
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Precompute coverage
    block_to_tsubs = []
    block_to_tsubs_set = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        block_to_tsubs_set.append(set(covered))
        
    # Precompute inverse
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Precompute overlap matrix between blocks (how many t-subsets they share)
    # For efficiency, we'll compute this on-the-fly during greedy
    
    # Helper: Greedy Construction with packing-aware scoring
    def greedy_construction(randomize_ties=False, packing_weight=0.0):
        uncovered = np.ones(num_tsubs, dtype=bool)
        k_choose_t = len(block_to_tsubs[0])
        gains = np.full(num_blocks, k_choose_t, dtype=np.int32)
        
        selected_block_indices = []
        selected_set = set()
        
        while uncovered.any():
            if time.time() > deadline:
                return None
            
            # Compute effective score: gain - packing_weight * overlap_with_selected
            if packing_weight > 0 and selected_set:
                # For each block, compute overlap with already selected blocks
                scores = np.zeros(num_blocks, dtype=np.float64)
                for b in range(num_blocks):
                    if gains[b] == 0:
                        scores[b] = -1
                        continue
                    overlap = 0
                    for s_idx in block_to_tsubs[b]:
                        if not uncovered[s_idx]:
                            # This t-subset is already covered, check if by selected blocks
                            # We need to know how many selected blocks cover this t-subset
                            pass
                    # Simplified: just use gain for now, packing is handled in local search
                    scores[b] = gains[b]
            else:
                scores = gains.astype(np.float64)
            
            if randomize_ties and random.random() < 0.3:
                top_indices = np.argsort(scores)[-15:]
                b = random.choice(top_indices)
            else:
                b = int(np.argmax(scores))
            
            if gains[b] == 0:
                break
            
            selected_block_indices.append(b)
            selected_set.add(b)
            
            for s_idx in block_to_tsubs[b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        if uncovered.any():
            return None
            
        return selected_block_indices

    # Helper: Local Optimization with faster candidate search
    def local_optimize(blocks, max_iters=100):
        if not blocks:
            return []
            
        selected_set = set(blocks)
        selected_list = list(blocks)
        
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        def is_redundant(b_idx):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        def find_replacement_fast(unique_tsubs, exclude_set):
            if not unique_tsubs:
                return None
            
            first_s = unique_tsubs[0]
            candidates = tsub_to_blocks_np[first_s]
            
            best_candidate = None
            best_score = -1
            
            for c in candidates:
                if c in exclude_set:
                    continue
                
                # Check if c covers all unique t-subsets
                c_set = block_to_tsubs_set[c]
                covers_all = True
                for s in unique_tsubs:
                    if s not in c_set:
                        covers_all = False
                        break
                
                if covers_all:
                    # Score: prefer blocks that cover more already-covered t-subsets (less waste)
                    waste = 0
                    for s in block_to_tsubs[c]:
                        if cover_count[s] > 0:
                            waste += 1
                    score = -waste
                    if score > best_score:
                        best_score = score
                        best_candidate = c
            
            return best_candidate
        
        improved = True
        iter_count = 0
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Step A: Redundancy Removal (multiple passes)
            redundant_blocks = []
            for b in selected_list:
                if is_redundant(b):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_list.remove(b)
                    selected_set.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step B: Swap with better candidate selection
            unique_tsubs_map = {}
            unique_counts = {}
            
            for b in selected_list:
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
                b_in = find_replacement_fast(s_uniq, selected_set)
                
                if b_in is not None:
                    selected_list.remove(b_out)
                    selected_set.remove(b_out)
                    selected_list.append(b_in)
                    selected_set.add(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break
            
            if not swap_made:
                break
                
        return selected_list

    best_solution = None
    best_len = float('inf')
    
    # Multi-start with varying strategies
    num_starts = 30
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Vary randomization and packing weight
        if i == 0:
            blocks = greedy_construction(randomize_ties=False, packing_weight=0.0)
        elif i % 3 == 0:
            blocks = greedy_construction(randomize_ties=True, packing_weight=0.0)
        else:
            blocks = greedy_construction(randomize_ties=True, packing_weight=0.0)
        
        if blocks is None:
            continue
        
        optimized_blocks = local_optimize(blocks, max_iters=50)
        
        if len(optimized_blocks) < best_len:
            best_len = len(optimized_blocks)
            best_solution = optimized_blocks
            
        if best_len <= 151:
            break

    if best_solution is None:
        blocks = greedy_construction(randomize_ties=False)
        if blocks:
            best_solution = local_optimize(blocks)
        else:
            return []
            
    result = [list(all_k_subsets[i]) for i in best_solution]
    return result
