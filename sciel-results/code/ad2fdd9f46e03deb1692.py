import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate all 4-subsets of range(11)
    all_subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(all_subsets)
    
    # Precompute bitmasks
    masks = [0] * n_subsets
    for i, s in enumerate(all_subsets):
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks[i] = m
        
    # Precompute popcount for 11-bit numbers
    max_val = 1 << D
    popcount = [0] * max_val
    for i in range(1, max_val):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    best_size = 0
    best_indices = []
    
    # Greedy with random restarts
    # Use time limit to determine number of restarts
    while time.time() < deadline:
        # Shuffle indices
        order = list(range(n_subsets))
        random.shuffle(order)
        
        chosen = []
        chosen_masks = []
        
        for idx in order:
            m = masks[idx]
            compatible = True
            for cm in chosen_masks:
                if popcount[m & cm] > 2:
                    compatible = False
                    break
            if compatible:
                chosen.append(idx)
                chosen_masks.append(m)
                
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_indices = chosen[:]
            
    # Construct vectors
    vectors = []
    for idx in best_indices:
        subset = all_subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx_pos in enumerate(subset):
                v[idx_pos] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
