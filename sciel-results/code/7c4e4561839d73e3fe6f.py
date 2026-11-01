import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Strategy 1: Attempt the high-density construction mentioned in the prompt
    # 1. Weight-4 vectors with even number of negative entries
    # 2. Weight-1 vectors with values +/- 2
    
    vectors = []
    
    # Generate weight-4 vectors
    # Support: 4 indices out of 11
    for support in itertools.combinations(range(D), 4):
        # Signs: 16 combinations, filter for even number of -1s
        for signs in itertools.product([1, -1], repeat=4):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                v = [0] * D
                for idx, val in zip(support, signs):
                    v[idx] = val
                vectors.append(v)
                
    # Generate weight-1 vectors with values +/- 2
    for i in range(D):
        v_pos = [0] * D
        v_pos[i] = 2
        vectors.append(v_pos)
        
        v_neg = [0] * D
        v_neg[i] = -2
        vectors.append(v_neg)
        
    arr = np.array(vectors, dtype=np.int64)
    
    # Check validity
    # Condition: min pairwise dist >= max norm
    # Squared dist >= Squared max norm
    
    if len(arr) > 0:
        norms_sq = np.sum(arr**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Calculate pairwise squared distances
        # Use broadcasting for efficiency, but memory might be an issue for 2662 vectors
        # 2662 x 2662 x 11 floats is ~ 77 MB. Acceptable.
        
        if time.time() < deadline:
            # Compute distance matrix
            # dist_sq[i, j] = ||v_i - v_j||^2
            # We only need to check if any dist_sq < max_norm_sq for i != j
            
            # Optimization: Check against max_norm_sq
            # If max_norm_sq is 4, we need dist_sq >= 4.
            
            # To save memory/time, we can check in chunks or just do it if N is small enough
            N = len(arr)
            if N <= 3000:
                # Compute all pairwise distances
                # v_i^2 + v_j^2 - 2 v_i . v_j
                # norms_sq is available
                dot_products = arr @ arr.T
                dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot_products
                
                # Set diagonal to infinity to ignore self-distance
                np.fill_diagonal(dist_sq, np.inf)
                
                min_dist_sq = np.min(dist_sq)
                
                if min_dist_sq >= max_norm_sq - 1e-9:
                    # Valid
                    return arr
                else:
                    # Invalid, fall back to smaller set
                    pass
            else:
                # If too large, assume invalid or check subset?
                # For now, assume invalid if too large to check quickly
                pass
    
    # Fallback: Weight-2 vectors (D11 minimal vectors)
    # These are known to be valid.
    vectors_fallback = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors_fallback.append(v)
                
    return np.array(vectors_fallback, dtype=np.int64)

# EVOLVE-BLOCK-END
