import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Precompute t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # Build block_bitmasks: for each block, bitmask of t-subsets it covers
    block_bitmasks = [0] * n_k_blocks
    block_points = all_k_blocks
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            mask |= (1 << idx)
        block_bitmasks[i] = mask
    
    # Build reverse index: for each t-subset, which blocks cover it
    t_sub_to_blocks = [[] for _ in range(n_t_subsets)]
    for i in range(n_k_blocks):
        m = block_bitmasks[i]
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            t_sub_to_blocks[idx].append(i)
            m ^= lsb
    
    full_mask = (1 << n_t_subsets) - 1
    
    def greedy_cover_scarcity():
        """Greedy with dynamic scarcity weighting"""
        uncovered_mask = full_mask
        selected_blocks = []
        
        # Precompute initial degree: how many blocks cover each t-subset
        degree = np.array([len(t_sub_to_blocks[j]) for j in range(n_t_subsets)], dtype=np.float64)
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_score = -1
            
            for i in range(n_k_blocks):
                mask_i = block_bitmasks[i]
                # Count uncovered t-subsets this block covers
                m = uncovered_mask & mask_i
                if m == 0:
                    continue
                
                # Compute scarcity-weighted score
                # For each uncovered t-subset in this block, add 1/degree
                score = 0.0
                mm = m
                while mm:
                    lsb = mm & -mm
                    idx = lsb.bit_length() - 1
                    d = degree[idx]
                    if d > 0:
                        score += 1.0 / d
                    mm ^= lsb
                
                # Normalize by count to get average scarcity
                cnt = m.bit_count()
                avg_scarcity = score / cnt if cnt > 0 else 0
                
                # Combine: prefer high coverage AND high scarcity
                combined = cnt * 10 + avg_scarcity * 5
                
                if combined > best_score:
                    best_score = combined
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
            
            # Update degrees for newly covered t-subsets
            m = block_bitmasks[best_idx] & ~uncovered_mask  # t-subsets just covered
            mm = m
            while mm:
                lsb = mm & -mm
                idx = lsb.bit_length() - 1
                degree[idx] -= 1
                mm ^= lsb
        
        return selected_blocks
    
    def greedy_cover_plain():
        """Standard greedy: maximize uncovered t-subsets covered"""
        uncovered_mask = full_mask
        selected_blocks = []
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_idx = -1
            best_gain = 0
            
            for i in range(n_k_blocks):
                gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1:
                break
                
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_bitmasks[best_idx]
        
        return selected_blocks
    
    def local_search_remove(blocks_indices):
        """Remove redundant blocks"""
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        
        for b_idx in blocks_indices:
            m = block_bitmasks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            
            for b_idx in blocks_indices:
                m = block_bitmasks[b_idx]
                essential = False
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        essential = True
                        break
                    m ^= lsb
                if not essential:
                    to_remove.append(b_idx)
            
            if to_remove:
                remove_set = set(to_remove)
                blocks_indices = [b for b in blocks_indices if b not in remove_set]
                for b_idx in to_remove:
                    m = block_bitmasks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                changed = True
                
        return blocks_indices
    
    def local_search_swap(blocks_indices):
        """Try removing blocks and finding replacements via reverse index"""
        if time.time() > deadline:
            return blocks_indices
        
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in blocks_indices:
            m = block_bitmasks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        
        selected_set = set(blocks_indices)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            for i, b_idx in enumerate(list(blocks_indices)):
                mask_remove = block_bitmasks[b_idx]
                m = mask_remove
                uncovered_after_remove = []
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        uncovered_after_remove.append(idx)
                    m ^= lsb
                
                if not uncovered_after_remove:
                    blocks_indices = blocks_indices[:i] + blocks_indices[i+1:]
                    m = mask_remove
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    selected_set.discard(b_idx)
                    improved = True
                    break
                
                # Find a block that covers all uncovered_after_remove
                target_set = set(uncovered_after_remove)
                best_swap = -1
                best_extra = float('inf')
                
                # Use reverse index: find blocks covering the first uncovered t-subset
                candidates = t_sub_to_blocks[uncovered_after_remove[0]]
                
                for j in candidates:
                    if j in selected_set:
                        continue
                    mask_j = block_bitmasks[j]
                    # Check if covers all uncovered
                    m = mask_j
                    covers_all = True
                    mm = mask_j
                    for idx in uncovered_after_remove:
                        if not (mm & (1 << idx)):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # Count extra coverage (t-subsets not in target)
                    extra = (mask_j & ~full_mask).bit_count()  # simplified
                    # Better: count t-subsets in mask_j that are not in target_set
                    extra_cnt = 0
                    mm = mask_j
                    while mm:
                        lsb = mm & -mm
                        idx = lsb.bit_length() - 1
                        if idx not in target_set:
                            extra_cnt += 1
                        mm ^= lsb
                    
                    if extra_cnt < best_extra:
                        best_extra = extra_cnt
                        best_swap = j
                
                if best_swap != -1:
                    blocks_indices[i] = best_swap
                    selected_set.add(best_swap)
                    selected_set.discard(b_idx)
                    
                    m = block_bitmasks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    
                    m = block_bitmasks[best_swap]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    
                    improved = True
                    break
        
        return blocks_indices
    
    def cover_holes(holes_mask):
        new_blocks = []
        uncovered = holes_mask
        while uncovered:
            if time.time() > deadline:
                break
            best_idx = -1
            best_gain = 0
            for i in range(n_k_blocks):
                gain = (uncovered & block_bitmasks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            new_blocks.append(best_idx)
            uncovered &= ~block_bitmasks[best_idx]
        return new_blocks
    
    # Run both greedy strategies and pick the better
    scarcity_blocks = greedy_cover_scarcity()
    plain_blocks = greedy_cover_plain()
    
    # Clean both
    scarcity_blocks = local_search_remove(scarcity_blocks)
    plain_blocks = local_search_remove(plain_blocks)
    
    if len(scarcity_blocks) <= len(plain_blocks):
        best_blocks = scarcity_blocks
    else:
        best_blocks = plain_blocks
    
    best_blocks = local_search_swap(best_blocks)
    best_count = len(best_blocks)
    
    # Perturbation loop
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        perturbed_blocks = list(best_blocks)
        
        num_remove = min(len(perturbed_blocks), random.randint(3, 8))
        if num_remove == 0:
            break
        to_remove = random.sample(perturbed_blocks, num_remove)
        remove_set = set(to_remove)
        perturbed_blocks = [b for b in perturbed_blocks if b not in remove_set]
        
        covered_mask = 0
        for b_idx in perturbed_blocks:
            covered_mask |= block_bitmasks[b_idx]
            
        holes_mask = full_mask & ~covered_mask
        added_blocks = cover_holes(holes_mask)
        
        new_total_blocks = perturbed_blocks + added_blocks
        new_count = len(new_total_blocks)
        
        if new_count < best_count:
            cleaned_blocks = local_search_remove(new_total_blocks)
            cleaned_blocks = local_search_swap(cleaned_blocks)
            if len(cleaned_blocks) < best_count:
                best_blocks = cleaned_blocks
                best_count = len(best_blocks)
            else:
                if new_count < best_count:
                    best_blocks = new_total_blocks
                    best_count = len(best_blocks)
        
        # Biased greedy restart
        if time.time() < deadline - 10 and len(best_blocks) > 10:
            bias_size = max(5, len(best_blocks) // 4)
            biased = random.sample(best_blocks, bias_size)
            biased_set = set(biased)
            
            # Greedy with bias
            uncovered_mask = full_mask
            biased_result = []
            while uncovered_mask:
                if time.time() > deadline:
                    break
                best_idx = -1
                best_gain = -1
                for i in range(n_k_blocks):
                    gain = (uncovered_mask & block_bitmasks[i]).bit_count()
                    if i in biased_set:
                        gain += 0.1
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                biased_result.append(best_idx)
                uncovered_mask &= ~block_bitmasks[best_idx]
            
            biased_result = local_search_remove(biased_result)
            biased_result = local_search_swap(biased_result)
            if len(biased_result) < best_count:
                best_blocks = biased_result
                best_count = len(best_blocks)
    
    # Final polish
    if time.time() < deadline - 2:
        best_blocks = local_search_remove(best_blocks)
        best_blocks = local_search_swap(best_blocks)
    
    final_blocks = [list(block_points[i]) for i in best_blocks]
    return final_blocks
