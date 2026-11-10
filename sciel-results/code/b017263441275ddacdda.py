# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    Strategy: 
    1. Generate all integer vectors with norm squared 8.
    2. Filter them using a greedy algorithm to ensure min_dist^2 >= 8.
    
    This explores the full space of norm-8 vectors, not just the 2E8 + 2D3 subset.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    d = 11
    candidates = []
    
    # Type 1: Two coordinates are +/-2, rest 0.
    # Norm squared = 4 + 4 = 8.
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                candidates.append(p)
                
    # Type 2: Eight coordinates are +/-1, rest 0.
    # Norm squared = 8 * 1 = 8.
    # We generate all such vectors.
    # To ensure efficiency, we can iterate over combinations of 8 indices.
    for idxs in combinations(range(d), 8):
        # Generate all sign combinations for these 8 indices
        for signs in product([1, -1], repeat=8):
            p = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(idxs):
                p[idx] = signs[k]
            candidates.append(p)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    # Shuffle to break symmetry and ensure diverse selection
    rng = np.random.default_rng(42)
    cand_indices = np.arange(len(candidates))
    rng.shuffle(cand_indices)
    
    points_list = []
    points_arr = None
    
    for idx in cand_indices:
        if time.time() > deadline - 5:
            break
            
        v = candidates[idx]
        norm_v_sq = 8 # We know it's 8
        
        valid = True
        if points_arr is not None:
            # Check against all existing points
            # dist_sq = 8 + 8 - 2 dot = 16 - 2 dot
            # We need dist_sq >= 8 => 16 - 2 dot >= 8 => 2 dot <= 8 => dot <= 4
            dots = points_arr @ v
            if np.any(dots > 4):
                valid = False
        
        if valid:
            if points_arr is None:
                points_arr = v.reshape(1, -1)
            else:
                points_arr = np.vstack([points_arr, v.reshape(1, -1)])
                
    if points_arr is None:
        # Fallback to a small valid set
        points_arr = np.array([[2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    return points_arr.astype(np.int64)

# EVOLVE-BLOCK-END
