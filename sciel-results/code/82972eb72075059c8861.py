import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Generate all vectors with exactly two non-zero entries (+/-1)
    vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    # Convert to numpy array
    vecs = np.array(vectors, dtype=np.int64)
    
    # Verify the condition: min pairwise distance >= max norm
    # This is O(N^2 * D)
    # N = 220, so 220^2 * 11 ~ 500,000 ops. Very fast.
    
    if len(vecs) == 0:
        return vecs
        
    norms = np.linalg.norm(vecs, axis=1)
    max_norm = np.max(norms)
    
    # Calculate pairwise distances
    # Using broadcasting
    # dist^2 = ||u||^2 + ||v||^2 - 2 u.v
    dots = vecs @ vecs.T
    dist_sq = norms[:, None]**2 + norms[None, :]**2 - 2 * dots
    
    # Check min distance
    # Exclude diagonal
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    
    if min_dist_sq < max_norm**2 - 1e-9:
        # This should not happen for this construction
        # But if it does, we might need to prune
        pass
        
    return vecs

# EVOLVE-BLOCK-START
# The code above is the core logic. 
# I will wrap it in the required structure.
# EVOLVE-BLOCK-END
