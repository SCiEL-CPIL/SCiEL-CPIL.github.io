"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Generate all integer vectors with exactly two non-zero entries of value +/-1
    # These have norm sqrt(2).
    # Minimum distance between any two distinct such vectors is sqrt(2) (if they share one index)
    # or 2 (if they are disjoint).
    # Max norm is sqrt(2).
    # Condition: min_dist >= max_norm => sqrt(2) >= sqrt(2). Holds.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    # Check time
    if time.time() > deadline:
        return np.array(vectors, dtype=np.int64)
        
    # The set of 220 vectors is valid.
    # Can we add more?
    # Attempting to add vectors of norm 1 (e.g., [1,0,...]) fails because distance to [1,1,...] is 1 < sqrt(2).
    # Attempting to add vectors of norm sqrt(3) fails because distance to [1,1,0...] is 1 < sqrt(3) (max norm would be sqrt(3)).
    
    # Thus, 220 is likely the maximum for this specific "integer" constraint if we stick to small norms.
    # There is no simple way to jump to 230 with integer vectors without a specific lattice structure that yields integers.
    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
