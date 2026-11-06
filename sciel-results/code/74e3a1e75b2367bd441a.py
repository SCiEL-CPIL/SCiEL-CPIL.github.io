import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    points = range(12)
    blocks = list(combinations(points, 6))
    n_blocks = len(blocks)
    
    all_5subs = list(combinations(points, 5))
    n_5subs = len(all_5subs)
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    
    block_masks = [0] * n_blocks
    for i, b in enumerate(blocks):
        mask = 0
        for s in combinations(b, 5):
            idx = sub_to_idx[s]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    bit_to_blocks = [[] for _ in range(n_5subs)]
    for b_idx in range(n_blocks):
        mask = block_masks[b_idx]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bit_to_blocks[idx].append(b_idx)
            m ^= lsb

    full_mask = (1 << n_5subs) - 1
    best_blocks = []
    best_count = 0
    found_optimal = False
    
    def backtrack(covered_mask, selected):
        nonlocal best_blocks, best_count, found_optimal
        
        if time.time() > deadline:
            return False
            
        if len(selected) == 132:
            best_blocks = selected[:]
            best_count = 132
            found_optimal = True
            return True
            
        # Pruning: Max possible blocks
        # Count uncovered bits
        # Popcount of (full_mask ^ covered_mask)
        remaining_bits = (full_mask ^ covered_mask).bit_count()
        if len(selected) + remaining_bits // 6 <= best_count:
            return False
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_blocks = selected[:]
            
        # MRV
        min_count = float('inf')
        min_candidates = []
        
        for bit_idx in range(n_5subs):
            if not (covered_mask & (1 << bit_idx)):
                candidates = bit_to_blocks[bit_idx]
                valid_candidates = []
                for b_idx in candidates:
                    if not (covered_mask & block_masks[b_idx]):
                        valid_candidates.append(b_idx)
                        
                count = len(valid_candidates)
                if count < min_count:
                    min_count = count
                    min_candidates = valid_candidates
                    if count == 0:
                        break
                        
        if min_count == 0:
            return False
            
        # Deterministic order: sort by block index to be stable
        # min_candidates are already in the order they appear in bit_to_blocks
        # which is block index order.
        
        for b_idx in min_candidates:
            selected.append(b_idx)
            new_covered = covered_mask | block_masks[b_idx]
            
            if backtrack(new_covered, selected):
                return True
                
            selected.pop()
            
        return False

    backtrack(0, [])
    
    return [list(blocks[i]) for i in best_blocks]
