import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k = len(k_subsets)
    
    # Precompute mask for each block as Python int
    masks_int = [0] * num_k
    # Precompute list of t-subset indices for each block
    block_indices = [None] * num_k
    
    for i, block in enumerate(k_subsets):
        mask = 0
        inds = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            inds.append(idx)
        masks_int[i] = mask
        block_indices[i] = inds
        
    if time.time() > deadline:
        return []

    best_blocks = None
    best_count = float('inf')
    
    # Randomized Greedy with Reduction
    # We can try multiple restarts
    
    # To speed up gain calculation, we can use a small pool of top candidates?
    # No, full scan is O(N).
    
    num_restarts = 50 # Adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(restart) # Reproducible per restart? Or use global random
        
        uncovered = (1 << num_t) - 1
        blocks_in_cover = []
        
        while uncovered:
            # Calculate gains
            # Use list comprehension
            gains = [(m & uncovered).bit_count() for m in masks_int]
            
            max_gain = max(gains)
            if max_gain == 0:
                break
                
            # Pick random from max gainers
            # To avoid building full list if max_gain is rare, 
            # we can just pick first max? 
            # Randomizing among max gainers is good.
            # But building list is O(N).
            
            # Optimization: 
            # If we just pick the first max, it's deterministic.
            # To randomize, we can add small random noise to gains?
            # gains_noisy = [g + random.random()*0.1 for g in gains]
            # This breaks ties randomly.
            # But we want to pick from top K, not just max.
            
            # Let's stick to picking from max gainers for simplicity and speed.
            # It's a valid randomized greedy variant.
            
            # Find one max gainer randomly
            # Instead of list comprehension for candidates, 
            # we can do:
            # candidates = []
            # for i, g in enumerate(gains):
            #     if g == max_gain:
            #         candidates.append(i)
            # chosen = random.choice(candidates)
            
            # This is O(N).
            
            # Alternative: 
            # idx = random.randint(0, num_k-1)
            # if gains[idx] < max_gain:
            #    # reject? No, we need max.
            
            # Let's just use the list comprehension for candidates.
            candidates = [i for i, g in enumerate(gains) if g == max_gain]
            chosen = random.choice(candidates)
            
            uncovered &= ~masks_int[chosen]
            blocks_in_cover.append(chosen)
            
        # Reduction
        # Use counts array
        counts = np.zeros(num_t, dtype=np.uint16)
        for idx in blocks_in_cover:
            for ind in block_indices[idx]:
                counts[ind] += 1
                
        changed = True
        while changed:
            changed = False
            # Iterate in reverse
            for i in range(len(blocks_in_cover) - 1, -1, -1):
                idx = blocks_in_cover[i]
                redundant = True
                for ind in block_indices[idx]:
                    if counts[ind] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ind in block_indices[idx]:
                        counts[ind] -= 1
                    blocks_in_cover.pop(i)
                    changed = True
                    break # Restart scan
                    
        if len(blocks_in_cover) < best_count:
            best_count = len(blocks_in_cover)
            best_blocks = blocks_in_cover.copy()
            
    if best_blocks is None:
        return []
        
    result = [list(k_subsets[idx]) for idx in best_blocks]
    return result
