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
    
    # 2. Greedy Construction with proper tracking
    uncovered = (1 << n_subsets) - 1
    selected_blocks = []
    selected_set = set()
    cover_count = [0] * n_subsets
    
    while uncovered:
        if time.time() > deadline - 5:
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
    while changed and time.time() < deadline - 2:
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
                
    # 4. Final Check and Return
    final_blocks = [list(k_blocks[i]) for i in selected_blocks]
    return final_blocks
