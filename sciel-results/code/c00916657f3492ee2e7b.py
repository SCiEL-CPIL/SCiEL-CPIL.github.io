import numpy as np
import itertools
import random
import time

D = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    
    # Precompute popcount for all 2^11 masks
    max_mask = 1 << D
    popcount = [0] * max_mask
    for i in range(1, max_mask):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Generate all 4-subsets of {0..10}
    subsets = list(itertools.combinations(range(D), 4))
    masks = [0] * len(subsets)
    for i, s in enumerate(subsets):
        m = 0
        for j in s:
            m |= (1 << j)
        masks[i] = m
        
    n_subsets = len(subsets)
    
    best_size = 0
    best_supports = []
    
    # Greedy with random restarts
    # Use a fixed seed for reproducibility in tests? No, random is fine.
    # We will run as many restarts as possible.
    
    # To speed up, we can check time every 100 restarts.
    restarts_done = 0
    while time.time() < deadline - 2.0:
        order = list(range(n_subsets))
        random.shuffle(order)
        
        current_supports = []
        
        for idx in order:
            m = masks[idx]
            compatible = True
            # Check against all in current_supports
            for m2 in current_supports:
                if popcount[m & m2] >= 3:
                    compatible = False
                    break
            if compatible:
                current_supports.append(m)
                
        if len(current_supports) > best_size:
            best_size = len(current_supports)
            best_supports = current_supports[:]
            
        restarts_done += 1
        if restarts_done % 100 == 0:
            # Optional: print progress to stderr if needed, but not required.
            pass

    # Construct vectors
    vectors = []
    for m in best_supports:
        # Get indices
        indices = [i for i in range(D) if (m >> i) & 1]
        # Generate all 2^4 = 16 sign combinations
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, idx in enumerate(indices):
                v[idx] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
