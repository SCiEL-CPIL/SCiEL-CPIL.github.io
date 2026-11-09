import numpy as np
import time
import random
from itertools import combinations

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # Generate all candidates
    candidates = []
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                candidates.append(p)
                
    # candidates is a list of 220 arrays
    n_cands = len(candidates)
    
    # Precompute conflict matrix? Or check on fly?
    # Checking on fly is O(k) per candidate. k is small.
    
    def is_valid_addition(new_point, current_set):
        for p in current_set:
            # Check if new_point and p share exactly one index with same sign
            # Non-zero indices
            nz_new = np.nonzero(new_point)[0]
            nz_p = np.nonzero(p)[0]
            
            # Since all have exactly 2 non-zeros
            # Intersection size
            inter = set(nz_new) & set(nz_p)
            if len(inter) == 1:
                idx = list(inter)[0]
                if new_point[idx] == p[idx]:
                    return False
            # If len(inter) == 2, they are in same group, distance is 4 or 8, always valid
            # If len(inter) == 0, distance is 4, valid
        return True

    best_set = []
    
    # Try a few random orders
    for restart in range(1000):
        if time.time() > deadline:
            break
            
        random.shuffle(candidates)
        current_set = []
        
        for p in candidates:
            if is_valid_addition(p, current_set):
                current_set.append(p)
                
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            if len(best_set) >= 22: # Theoretical max
                break
                
    # Convert to array
    if not best_set:
        # Fallback to seed
        points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
        return points
        
    points = np.array(best_set, dtype=np.int64)
    return points
