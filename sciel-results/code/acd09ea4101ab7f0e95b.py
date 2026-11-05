import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Pre-computation
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(k_blocks)
    
    # For each block, compute the list of t-subset indices it covers
    # and a bitset mask
    block_masks = [0] * n_blocks
    block_subsets = [[] for _ in range(n_blocks)]
    
    for b_idx, block in enumerate(k_blocks):
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            subs.append(idx)
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        block_subsets[b_idx] = subs
    
    # 2. Greedy Construction
    uncovered = (1 << n_subsets) - 1
    selected_blocks = []
    selected_set = set()
    cover_count = [0] * n_subsets
    
    while uncovered:
        if time.time() > deadline - 25:  # Leave time for local search
            break
            
        best_block = -1
        best_gain = 0
        best_mask = 0
        
        # Find block with max intersection with uncovered
        for b_idx in range(n_blocks):
            if b_idx in selected_set:
                continue
            gain = (block_masks[b_idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = b_idx
                best_mask = block_masks[b_idx]
                
        if best_block == -1:
            break
            
        selected_blocks.append(best_block)
        selected_set.add(best_block)
        uncovered &= ~best_mask
        
        for s in block_subsets[best_block]:
            cover_count[s] += 1
            
    # 3. Redundancy Removal - multiple passes
    changed = True
    while changed and time.time() < deadline - 8:
        changed = False
        for i in range(len(selected_blocks) - 1, -1, -1):
            b_idx = selected_blocks[i]
            redundant = True
            for s in block_subsets[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            
            if redundant:
                for s in block_subsets[b_idx]:
                    cover_count[s] -= 1
                selected_blocks.pop(i)
                selected_set.discard(b_idx)
                changed = True
                
    # 4. Local Search: 1-opt removals and 2-opt swaps
    if time.time() < deadline - 2:
        # Build reverse index: for each t-subset, which selected blocks cover it
        subset_to_selected_blocks = [[] for _ in range(n_subsets)]
        for b_idx in selected_blocks:
            for s in block_subsets[b_idx]:
                subset_to_selected_blocks[s].append(b_idx)
        
        max_iterations = 200
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
            
            improved = False
            
            # Try 1-opt: remove a block if all its subsets are covered by others
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                redundant = True
                for s in block_subsets[b_idx]:
                    if len(subset_to_selected_blocks[s]) <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for s in block_subsets[b_idx]:
                        subset_to_selected_blocks[s].remove(b_idx)
                    selected_blocks.pop(i)
                    selected_set.discard(b_idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-opt: swap one selected block with one unselected block
            for i in range(len(selected_blocks)):
                if time.time() > deadline - 2:
                    break
                    
                b_remove = selected_blocks[i]
                # Find subsets that would become uncovered
                critical_subsets = [s for s in block_subsets[b_remove] 
                                    if len(subset_to_selected_blocks[s]) == 1]
                
                if not critical_subsets:
                    # Redundant, remove it
                    for s in block_subsets[b_remove]:
                        subset_to_selected_blocks[s].remove(b_remove)
                    selected_blocks.pop(i)
                    selected_set.discard(b_remove)
                    improved = True
                    break
                    
                # Build mask of critical subsets
                critical_mask = 0
                for s in critical_subsets:
                    critical_mask |= (1 << s)
                    
                # Find candidate replacement blocks (sample to keep fast)
                best_replace = -1
                best_score = float('-inf')
                
                unselected = [b for b in range(n_blocks) if b not in selected_set]
                if len(unselected) > 5000:
                    random.shuffle(unselected)
                    unselected = unselected[:5000]
                
                for b_add in unselected:
                    # Must cover all critical subsets
                    if (block_masks[b_add] & critical_mask) != critical_mask:
                        continue
                    
                    # Score: help barely-covered subsets, penalize adding redundancy
                    score = 0
                    for s in block_subsets[b_add]:
                        count = len(subset_to_selected_blocks[s])
                        if count == 1:
                            score += 2
                        elif count == 2:
                            score += 1
                        else:
                            score -= 1
                    
                    if score > best_score:
                        best_score = score
                        best_replace = b_add
                        
                if best_replace != -1:
                    # Perform swap
                    for s in block_subsets[b_remove]:
                        subset_to_selected_blocks[s].remove(b_remove)
                    for s in block_subsets[best_replace]:
                        subset_to_selected_blocks[s].append(best_replace)
                    selected_blocks[i] = best_replace
                    selected_set.discard(b_remove)
                    selected_set.add(best_replace)
                    improved = True
                    break
                    
            if not improved:
                break
    
    # 5. Final redundancy removal
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(len(selected_blocks) - 1, -1, -1):
            b_idx = selected_blocks[i]
            redundant = True
            for s in block_subsets[b_idx]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            
            if redundant:
                for s in block_subsets[b_idx]:
                    cover_count[s] -= 1
                selected_blocks.pop(i)
                selected_set.discard(b_idx)
                changed = True
                
    # 6. Final Check and Return
    final_blocks = [list(k_blocks[i]) for i in selected_blocks]
    return final_blocks
