import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = 100.0  # Leave 20s margin
    
    best_vectors = []
    best_score = 0
    
    # We will try different weights (number of non-zero entries)
    # The vectors will be in {-1, 0, 1}^D
    # We assume all vectors in the final set have the same weight to simplify the norm constraint.
    # If we mix weights, the max norm increases, tightening the distance constraint.
    # Usually, optimal packing for this type of problem (spherical codes) uses constant norm.
    
    for w in range(2, D + 1):
        if time.time() - start_time > deadline:
            break
            
        # Generate all vectors of weight w
        # Choose w indices out of D
        # Assign +/- 1 to these indices
        candidates = []
        for indices in itertools.combinations(range(D), w):
            # Generate all sign combinations for these w indices
            # 2^w combinations
            for mask in range(1 << w):
                v = [0] * D
                for i, idx in enumerate(indices):
                    if (mask >> i) & 1:
                        v[idx] = 1
                    else:
                        v[idx] = -1
                candidates.append(v)
        
        candidates = np.array(candidates, dtype=np.int64)
        
        # Shuffle candidates
        np.random.shuffle(candidates)
        
        selected = []
        selected_arr = np.empty((0, D), dtype=np.int64)
        
        threshold = w / 2.0
        
        for v in candidates:
            if time.time() - start_time > deadline:
                break
                
            if len(selected_arr) == 0:
                selected_arr = v.reshape(1, -1)
                continue
                
            # Check compatibility
            # v . s <= threshold for all s in selected_arr
            dots = selected_arr @ v
            if np.max(dots) <= threshold:
                selected_arr = np.vstack([selected_arr, v.reshape(1, -1)])
        
        if len(selected_arr) > best_score:
            best_score = len(selected_arr)
            best_vectors = selected_arr
            
        # Optional: Early stopping if we hit a theoretical bound or time is tight
        # Theoretical bound for kissing number is not known exactly, but 2^D is a loose upper bound for this specific code?
        # Actually, for angular distance 60 deg, the bound is related to the size of the code.
        
    return best_vectors
