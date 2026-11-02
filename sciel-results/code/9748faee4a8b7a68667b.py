"""Covering design C(19,8,5) - Global Bitmask Greedy with Local Search + 2-exchange."""
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)
    
    # Map subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Precompute bitmask for each block
    block_masks = [0] * n_blocks
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    # Precompute bit positions for each block mask (list of t-subset indices covered)
    block_bit_lists = []
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        bits = []
        m = mask
        while m:
            idx = (m & -m).bit_length() - 1
            bits.append(idx)
            m &= m - 1
        block_bit_lists.append(bits)
    
    # Precompute sets for faster membership checks
    block_bit_sets = [set(bits) for bits in block_bit_lists]

    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    max_restarts = 300
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subs) - 1
        selected_blocks = []
        active_blocks = list(range(n_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_block_candidates = []
            
            new_active_blocks = []
            for b_idx in active_blocks:
                mask = block_masks[b_idx]
                gain = (mask & uncovered_mask).bit_count()
                if gain > 0:
                    new_active_blocks.append(b_idx)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_candidates = [b_idx]
                    elif gain == best_gain:
                        best_block_candidates.append(b_idx)
                        
            if not best_block_candidates:
                break
                
            chosen_block = rng.choice(best_block_candidates)
            selected_blocks.append(chosen_block)
            uncovered_mask &= ~block_masks[chosen_block]
            active_blocks = new_active_blocks
            
            if not active_blocks:
                break

        if uncovered_mask == 0:
            # Local Search Phase 1: Delete redundant blocks
            cover_counts = [0] * n_subs
            for b_idx in selected_blocks:
                for idx in block_bit_lists[b_idx]:
                    cover_counts[idx] += 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    is_redundant = True
                    for idx in block_bit_lists[b_idx]:
                        if cover_counts[idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        for idx in block_bit_lists[b_idx]:
                            cover_counts[idx] -= 1
                        selected_blocks.pop(i)
                        changed = True
                        break

            # Local Search Phase 2: 1-for-1 swap
            swap_improved = True
            max_swap_iters = 8
            for _ in range(max_swap_iters):
                if time.time() > deadline:
                    break
                swap_improved = False
                for i in range(len(selected_blocks)):
                    if time.time() > deadline:
                        break
                    b_idx = selected_blocks[i]
                    unique_to_block = []
                    for idx in block_bit_lists[b_idx]:
                        if cover_counts[idx] == 1:
                            unique_to_block.append(idx)
                    
                    if not unique_to_block:
                        for idx in block_bit_lists[b_idx]:
                            cover_counts[idx] -= 1
                        selected_blocks.pop(i)
                        swap_improved = True
                        break
                    
                    unique_set = set(unique_to_block)
                    n_tries = min(30, n_blocks - len(selected_blocks))
                    for _ in range(n_tries):
                        if time.time() > deadline:
                            break
                        cand = rng.randint(0, n_blocks - 1)
                        if cand in selected_blocks:
                            continue
                        if not unique_set.issubset(block_bit_sets[cand]):
                            continue
                        
                        # Simulate swap
                        new_cc = cover_counts[:]
                        for idx in block_bit_lists[b_idx]:
                            new_cc[idx] -= 1
                        for idx in block_bit_lists[cand]:
                            new_cc[idx] += 1
                        
                        valid = True
                        for idx in range(n_subs):
                            if new_cc[idx] < 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        
                        potential_deletions = 0
                        for j, sj in enumerate(selected_blocks):
                            if j == i:
                                continue
                            is_red = True
                            for idx in block_bit_lists[sj]:
                                if new_cc[idx] == 1:
                                    is_red = False
                                    break
                            if is_red:
                                potential_deletions += 1
                        
                        if potential_deletions > 0:
                            for idx in block_bit_lists[b_idx]:
                                cover_counts[idx] -= 1
                            for idx in block_bit_lists[cand]:
                                cover_counts[idx] += 1
                            selected_blocks[i] = cand
                            swap_improved = True
                            break
                        
                        if rng.random() < 0.05:
                            for idx in block_bit_lists[b_idx]:
                                cover_counts[idx] -= 1
                            for idx in block_bit_lists[cand]:
                                cover_counts[idx] += 1
                            selected_blocks[i] = cand
                            swap_improved = True
                            break
                    
                    if swap_improved:
                        break
                
                if not swap_improved:
                    break

            # Local Search Phase 3: 2-for-1 exchange (remove two blocks, add one)
            two_for_one_improved = True
            max_21_iters = 3
            for _ in range(max_21_iters):
                if time.time() > deadline:
                    break
                two_for_one_improved = False
                n_sel = len(selected_blocks)
                if n_sel < 2:
                    break
                for i in range(n_sel):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, n_sel):
                        if time.time() > deadline:
                            break
                        b1 = selected_blocks[i]
                        b2 = selected_blocks[j]
                        
                        # Compute what's uniquely covered by b1 or b2
                        # i.e., subsets where cover_count == 1 and covered by b1 or b2
                        unique_to_pair = set()
                        for idx in block_bit_lists[b1]:
                            if cover_counts[idx] == 1:
                                unique_to_pair.add(idx)
                        for idx in block_bit_lists[b2]:
                            if cover_counts[idx] == 1:
                                unique_to_pair.add(idx)
                        
                        if not unique_to_pair:
                            # Both are redundant
                            for idx in block_bit_lists[b1]:
                                cover_counts[idx] -= 1
                            for idx in block_bit_lists[b2]:
                                cover_counts[idx] -= 1
                            selected_blocks.pop(j)
                            selected_blocks.pop(i)
                            two_for_one_improved = True
                            break
                        
                        # Try to find a single block that covers all unique_to_pair
                        # and doesn't create negative counts
                        n_tries = min(50, n_blocks - len(selected_blocks))
                        for _ in range(n_tries):
                            if time.time() > deadline:
                                break
                            cand = rng.randint(0, n_blocks - 1)
                            if cand in selected_blocks:
                                continue
                            if not unique_to_pair.issubset(block_bit_sets[cand]):
                                continue
                            
                            # Simulate: remove b1, b2, add cand
                            new_cc = cover_counts[:]
                            for idx in block_bit_lists[b1]:
                                new_cc[idx] -= 1
                            for idx in block_bit_lists[b2]:
                                new_cc[idx] -= 1
                            for idx in block_bit_lists[cand]:
                                new_cc[idx] += 1
                            
                            valid = True
                            for idx in range(n_subs):
                                if new_cc[idx] < 0:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            
                            # Check for further deletions
                            potential_deletions = 0
                            for idx_ in range(len(selected_blocks)):
                                if idx_ == i or idx_ == j:
                                    continue
                                sj = selected_blocks[idx_]
                                is_red = True
                                for idx in block_bit_lists[sj]:
                                    if new_cc[idx] == 1:
                                        is_red = False
                                        break
                                if is_red:
                                    potential_deletions += 1
                            
                            # Net change: -2 + 1 - potential_deletions = -1 - potential_deletions
                            # Always improves by at least 1
                            if potential_deletions >= 0:
                                for idx in block_bit_lists[b1]:
                                    cover_counts[idx] -= 1
                                for idx in block_bit_lists[b2]:
                                    cover_counts[idx] -= 1
                                for idx in block_bit_lists[cand]:
                                    cover_counts[idx] += 1
                                # Remove b1 and b2 (remove higher index first)
                                selected_blocks.pop(j)
                                selected_blocks.pop(i)
                                selected_blocks.append(cand)
                                two_for_one_improved = True
                                break
                        
                        if two_for_one_improved:
                            break
                    
                    if two_for_one_improved:
                        break
                
                if not two_for_one_improved:
                    break

            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(all_blocks[i]) for i in selected_blocks]
                
                if best_count <= 335:
                    break

    if best_blocks is None:
        return []
        
    return best_blocks
