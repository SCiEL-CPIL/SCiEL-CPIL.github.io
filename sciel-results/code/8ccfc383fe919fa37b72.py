import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580
    
    # Precompute all 4-subsets and their 3-subset IDs
    # Map 3-subsets to IDs
    three_subsets = list(combinations(range(d), 3))
    three_to_id = {s: i for i, s in enumerate(three_subsets)}
    
    four_subsets = list(combinations(range(d), 4))
    # For each 4-subset, store the IDs of its 4 three-subsets
    four_to_threes = []
    for s4 in four_subsets:
        ids = []
        for i in range(4):
            # Remove i-th element to get a 3-subset
            s3 = tuple(x for j, x in enumerate(s4) if j != i)
            ids.append(three_to_id[s3])
        four_to_threes.append(np.array(ids, dtype=np.int64))
        
    four_to_threes = np.array(four_to_threes) # Shape (330, 4)
    
    best_supports = []
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Greedy packing search
    num_restarts = 0
    while time.time() < deadline - 10: # Leave 10s for construction
        num_restarts += 1
        order = rng.permutation(len(four_subsets))
        used = np.zeros(len(three_subsets), dtype=bool)
        selected = []
        
        for idx in order:
            ids = four_to_threes[idx]
            if not np.any(used[ids]):
                used[ids] = True
                selected.append(idx)
                
        if len(selected) > best_count:
            best_count = len(selected)
            best_supports = selected
            
        # If we hit the theoretical upper bound, stop
        if best_count >= 41:
            break
            
    # Construct points
    points = []
    for idx in best_supports:
        support = four_subsets[idx]
        # Generate all 16 sign combinations
        for signs in range(16):
            v = np.zeros(d, dtype=np.int64)
            for k, pos in enumerate(support):
                if (signs >> k) & 1:
                    v[pos] = 1
                else:
                    v[pos] = -1
            points.append(v)
            
    if len(points) < 2:
        # Fallback
        points = [np.ones(d, dtype=np.int64), -np.ones(d, dtype=np.int64)]
        
    pts_array = np.array(points, dtype=np.int64)
    return np.around(pts_array).astype(np.int64)
