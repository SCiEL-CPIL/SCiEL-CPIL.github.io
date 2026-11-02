import itertools
import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate all weight-4 vectors
    # There are 5280 of them
    candidates = []
    for idx in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, s in zip(idx, signs):
                v[pos] = s
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    
    # Greedy selection
    # Sort candidates to prioritize those that are "central" or just random?
    # Random shuffle might help avoid local minima, but greedy is deterministic.
    # Let's try a simple greedy: pick one, then pick next that is compatible with all picked.
    
    # To speed up, we can use a random order or multiple restarts.
    best_set = []
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
            
        order = rng.permutation(len(candidates))
        selected = []
        
        for i in order:
            if time.time() > deadline - 5:
                break
                
            c = candidates[i]
            if not selected:
                selected.append(c)
                continue
                
            # Check compatibility
            # dist2 >= 4  <=>  v_i . v_j <= 2
            # Compute dot products with all selected
            # selected is a list of arrays
            # Vectorized check
            dists = np.sum((selected - c) ** 2, axis=1)
            if np.min(dists) >= 4:
                selected.append(c)
                
        if len(selected) > len(best_set):
            best_set = selected
            if len(best_set) >= 504:
                break
                
    if not best_set:
        # Fallback to weight-2
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i], v[j] = si, sj
                    vectors.append(v)
        best_set = vectors
        
    return np.array(best_set, dtype=np.int64)
