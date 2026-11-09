import numpy as np
from itertools import combinations
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 590  # safety margin
    
    all_4sets = list(combinations(range(d), 4))
    n_sets = len(all_4sets)
    
    # Precompute bitmask for each 4-set
    masks = []
    for s in all_4sets:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask of indices j such that set i and set j share >= 3 indices
    conflict_masks = [0] * n_sets
    for i in range(n_sets):
        mi = masks[i]
        for j in range(n_sets):
            if i == j:
                continue
            inter = bin(mi & masks[j]).count('1')
            if inter >= 3:
                conflict_masks[i] |= (1 << j)
                
    best_supports = []
    rng = random.Random(42)
    
    # Run many random greedy trials
    num_tries = 2000
    for trial in range(num_tries):
        if time.time() > deadline - 10:
            break
            
        # Random ordering
        order = list(range(n_sets))
        rng.shuffle(order)
        
        selected = set()
        selected_mask = 0
        
        for idx in order:
            # Check if idx conflicts with any selected
            if (conflict_masks[idx] & selected_mask) == 0:
                selected.add(idx)
                selected_mask |= (1 << idx)
                
        if len(selected) > len(best_supports):
            best_supports = list(selected)
            
    # Generate points
    points = []
    for idx in best_supports:
        s = all_4sets[idx]
        i, j, k, l = s
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        points.append(v)
                        
    points_array = np.array(points, dtype=np.int64)
    return np.around(points_array).astype(np.int64)
