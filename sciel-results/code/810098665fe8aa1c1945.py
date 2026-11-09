import time
import random
from itertools import combinations

def construct():
    points = range(12)
    all_blocks = []
    block_masks = []
    
    # Precompute blocks and their 5-subset bitmasks
    for b in combinations(points, 6):
        masks = []
        for s in combinations(b, 5):
            mask = 0
            for p in s:
                mask |= (1 << p)
            masks.append(mask)
        all_blocks.append(list(b))
        block_masks.append(masks)
        
    n_blocks = len(all_blocks)
    start_time = time.time()
    deadline = 110.0 # 10s safety margin
    
    best_blocks = []
    best_count = 0
    
    # 1. Deterministic lexicographic (same as seed)
    # This is fast and establishes a baseline.
    covered = set()
    blocks = []
    for i in range(n_blocks):
        masks = block_masks[i]
        # Check if any 5-subset is already covered
        conflict = False
        for m in masks:
            if m in covered:
                conflict = True
                break
        if not conflict:
            blocks.append(all_blocks[i])
            covered.update(masks)
            if len(blocks) == 132:
                return blocks
    best_blocks = blocks
    best_count = len(blocks)
    
    # 2. Randomized restarts
    # If we already have 132, we returned. So best_count < 132 here.
    indices = list(range(n_blocks))
    
    while time.time() < deadline:
        random.shuffle(indices)
        
        covered = set()
        blocks = []
        for idx in indices:
            masks = block_masks[idx]
            conflict = False
            for m in masks:
                if m in covered:
                    conflict = True
                    break
            if not conflict:
                blocks.append(all_blocks[idx])
                covered.update(masks)
                if len(blocks) == 132:
                    return blocks
        
        if len(blocks) > best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    return best_blocks
