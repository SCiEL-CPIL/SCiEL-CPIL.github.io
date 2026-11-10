# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum squared norm is less than or equal to their minimum pairwise squared distance.
    
    Strategy:
    1. Generate all integer vectors with norm squared 4 (four +/-1s).
       Pool size: C(11,4) * 2^4 = 330 * 16 = 5280.
    2. Compatibility condition: u . v <= 2.
    3. Use randomized greedy search with multiple restarts to find a large compatible set.
    4. Use a pre-allocated numpy array for selected vectors to speed up dot products.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    # Generate all vectors with exactly four non-zero entries, each +1 or -1
    candidates = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    best_selected = []
    best_len = 0
    
    # Pre-allocate buffer for selected vectors in each restart
    # Max size is n_candidates
    sel_buf = np.zeros((n_candidates, d), dtype=np.int64)
    
    # Number of restarts
    # Each restart is O(N * K_max * d) with vectorized dot products.
    # N=5280, d=11.
    # If K_max ~ 1000, ops ~ 5280 * 1000 * 11 = 58M.
    # NumPy handles this in ~50ms.
    # 100 restarts ~ 5s.
    
    for restart in range(100):
        if time.time() > deadline - 10:
            break
            
        # Shuffle indices
        indices = np.random.permutation(n_candidates)
        
        k = 0
        for idx in indices:
            v = candidates[idx]
            
            # Check compatibility with all selected
            if k > 0:
                dots = sel_buf[:k] @ v
                if np.max(dots) > 2:
                    continue
            
            sel_buf[k] = v
            k += 1
        
        if k > best_len:
            best_len = k
            # Copy the best found so far
            best_selected = sel_buf[:k].copy()
            
    if not best_selected:
        # Fallback to a small valid set if something went wrong
        p1 = np.zeros(d, dtype=np.int64)
        p1[0] = 1
        p1[1] = 1
        p2 = np.zeros(d, dtype=np.int64)
        p2[0] = -1
        p2[1] = -1
        best_selected = np.array([p1, p2], dtype=np.int64)
        
    return best_selected

# EVOLVE-BLOCK-END
