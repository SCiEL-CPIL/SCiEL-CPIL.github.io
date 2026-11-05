import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    points = list(range(12))
    all_blocks = list(combinations(points, 6))
    n_blocks = len(all_blocks)
    
    # Precompute 5-subsets for each block
    # Map 5-subsets to indices for fast checking
    all_5subs = list(combinations(points, 5))
    sub_to_idx = {s: i for i, s in enumerate(all_5subs)}
    n_5subs = len(all_5subs)
    
    # block_subs[i] is a list of indices of 5-subsets in all_5subs
    block_subs = []
    for b in all_blocks:
        subs = [sub_to_idx[s] for s in combinations(b, 5)]
        block_subs.append(subs)
        
    best_blocks = []
    rng = random.Random(42)
    
    # Try many random restarts
    # Each restart is very fast
    for seed in range(1000):
        if time.time() > deadline:
            break
            
        if seed == 0:
            rng = random.Random(42)
        else:
            rng = random.Random(seed)
            
        indices = list(range(n_blocks))
        rng.shuffle(indices)
        
        used = [0] * n_5subs
        current_blocks = []
        
        for idx in indices:
            subs = block_subs[idx]
            conflict = False
            for s_idx in subs:
                if used[s_idx]:
                    conflict = True
                    break
            if not conflict:
                current_blocks.append(all_blocks[idx])
                for s_idx in subs:
                    used[s_idx] = 1
                    
                if len(current_blocks) == 132:
                    return current_blocks
        
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) == 132:
                return best_blocks
                
    return best_blocks
