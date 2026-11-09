import numpy as np
import time
from itertools import combinations

def generate_weight_vectors(d, weight):
    """
    Generates all integer vectors of length d with exactly 'weight' non-zero entries,
    each being +/- 1.
    Returns a list of numpy arrays.
    """
    vectors = []
    indices = list(range(d))
    
    # Generate all combinations of indices
    for combo in combinations(indices, weight):
        # Generate all sign combinations (2^weight)
        for mask in range(2 ** weight):
            v = np.zeros(d, dtype=np.int64)
            for k, idx in enumerate(combo):
                if (mask >> k) & 1:
                    v[idx] = 1
                else:
                    v[idx] = -1
            vectors.append(v)
            
    return vectors

def kissing_number11() -> np.ndarray:
    d = 11
    
    # The set of all weight-2 vectors in Z^11 is a valid kissing configuration.
    # Norm squared is 2.
    # Min distance squared is 2 (since max dot product is 1).
    # Condition: dist^2 >= norm^2 => 2 >= 2. Holds.
    
    w2_vectors = generate_weight_vectors(d, 2)
    points = np.array(w2_vectors, dtype=np.int64)
    
    # Optional: Verify validity to be safe (fast for 220 points)
    # k = len(points)
    # min_dist_sq = np.inf
    # max_norm_sq = 0
    # for i in range(k):
    #     n_i = np.dot(points[i], points[i])
    #     if n_i > max_norm_sq: max_norm_sq = n_i
    #     for j in range(i+1, k):
    #         diff = points[i] - points[j]
    #         d_sq = np.dot(diff, diff)
    #         if d_sq < min_dist_sq: min_dist_sq = d_sq
    # if min_dist_sq < max_norm_sq:
    #     # Fallback or error, but we know it's valid
    #     pass
    
    return points
