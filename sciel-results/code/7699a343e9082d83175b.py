import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = range(12)
    blocks = list(combinations(points, 6))
    n_blocks = len(blocks)
    
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    # Precompute masks
    block_masks = []
    for b in blocks:
        mask = 0
        for s in combinations(b, 5):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    # Precompute bit_to_blocks
    bit_to_blocks = [[] for _ in range(n_5subs)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bit_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    best_blocks = []
    best_count = 0
    
    # Symmetry Breaking: Fix the first block to be (0, 1, 2, 3, 4, 5)
    first_block_idx = blocks.index((0, 1, 2, 3, 4, 5))
    first_block_mask = block_masks[first_block_idx]
    
    def backtrack(covered_mask, selected_blocks, depth):
        nonlocal best_blocks, best_count
        
        if time.time() > deadline:
            return False
            
        if covered_mask == (1 << n_5subs) - 1:
            if len(selected_blocks) > best_count:
                best_blocks = selected_blocks[:]
                best_count = len(selected_blocks)
            return True
            
        remaining_mask = ((1 << n_5subs) - 1) ^ covered_mask
        if remaining_mask == 0:
            if len(selected_blocks) > best_count:
                best_blocks = selected_blocks[:]
                best_count = len(selected_blocks)
            return True

        # MRV: Find uncovered bit with fewest available blocks
        min_options = float('inf')
        chosen_bit = -1
        
        r = remaining_mask
        while r:
            lsb = r & -r
            bit_idx = lsb.bit_length() - 1
            
            avail_count = 0
            for b_idx in bit_to_blocks[bit_idx]:
                if not (covered_mask & block_masks[b_idx]):
                    avail_count += 1
                    if avail_count >= min_options:
                        break
            if avail_count < min_options:
                min_options = avail_count
                chosen_bit = bit_idx
                if min_options == 0:
                    return False
            r ^= lsb
            
        if chosen_bit == -1:
            return False
            
        candidates = bit_to_blocks[chosen_bit]
        valid_candidates = [b_idx for b_idx in candidates if not (covered_mask & block_masks[b_idx])]
        
        for b_idx in valid_candidates:
            selected_blocks.append(b_idx)
            if backtrack(covered_mask | block_masks[b_idx], selected_blocks, depth + 1):
                return True
            selected_blocks.pop()
                
        return False
        
    backtrack(first_block_mask, [first_block_idx], 0)
    
    if best_count >= 132:
        return [list(blocks[i]) for i in best_blocks]
        
    # Fallback: Greedy
    indices = list(range(n_blocks))
    covered = 0
    current_blocks = []
    for i in indices:
        if i == first_block_idx:
            continue
        m = block_masks[i]
        if not (covered & m):
            current_blocks.append(i)
            covered |= m
            
    if len(current_blocks) + 1 > best_count:
         best_blocks = [first_block_idx] + current_blocks
         best_count = len(best_blocks)

    return [list(blocks[i]) for i in best_blocks]
