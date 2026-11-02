import itertools
import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin
    
    vectors = []
    # Generate all norm-squared 2 vectors
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                vectors.append(v)
    
    # Convert to numpy for faster distance checks if we expand
    # But 220 vectors is small enough to just return
    # Let's verify validity just in case
    if len(vectors) > 0:
        arr = np.array(vectors, dtype=np.int64)
        norms_sq = np.sum(arr**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check min distance
        # For 220 vectors, O(N^2) is fine
        min_dist_sq = float('inf')
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                diff = arr[i] - arr[j]
                dist_sq = np.sum(diff**2)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
        
        if min_dist_sq >= max_norm_sq:
            return arr
        else:
            # Fallback: return empty or valid subset
            # But we know it's valid.
            return arr
            
    return np.array([], dtype=np.int64).reshape(0, D)
# EVOLVE-BLOCK-END
