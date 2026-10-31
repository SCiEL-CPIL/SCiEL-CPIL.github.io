import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Part 1: Scaled D11 minimal vectors
    # Two non-zeros +/- 2 in any 2 of 11 dimensions
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part 2: Generate all "Eight +/- 1, even parity" vectors in full D=11 space
    # This creates a large candidate pool
    candidates = []
    # Iterate over all subsets of size 8 from 11 dimensions
    for support in itertools.combinations(range(D), 8):
        # For each support, generate all sign assignments with even number of -1s
        # There are 2^8 = 256 assignments. Half have even parity.
        # We can just generate all and filter, or generate parity-constrained.
        for mask in range(256):
            minus_count = 0
            v = [0] * D
            for k in range(8):
                if (mask >> k) & 1:
                    v[support[k]] = -1
                    minus_count += 1
                else:
                    v[support[k]] = 1
            if minus_count % 2 == 0:
                candidates.append(v)
                
    # Greedy Selection
    # Start with Part 1
    selected = vectors
    # Convert to numpy for faster distance calculations if needed, 
    # but simple list of lists is fine for this size.
    
    # Target squared distance: R^2 = 8
    target_sq_dist = 8.0
    
    # To speed up, we can check candidates against selected.
    # We iterate through candidates.
    # If a candidate is valid, we add it to selected.
    
    # Optimization: 
    # The order of candidates might matter.
    # Randomizing or sorting by norm (all same) doesn't help much.
    # We just iterate.
    
    for v in candidates:
        # Check against all currently selected vectors
        valid = True
        for u in selected:
            # Calculate squared distance
            # dist_sq = sum((v[i]-u[i])**2 for i in range(D))
            # Using numpy for this single pair might be slower than pure python list comp?
            # Let's use numpy for the vector subtraction and norm
            diff = np.array(v) - np.array(u)
            if np.dot(diff, diff) < target_sq_dist:
                valid = False
                break
        if valid:
            selected.append(v)
            
    return np.array(selected, dtype=np.int64)
