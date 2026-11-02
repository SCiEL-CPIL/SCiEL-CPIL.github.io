import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    # Generate E8 roots scaled to have norm squared 8 in R^8, embedded in R^11
    # Type 1: Permutations of (2, 2, 0, ..., 0) with signs
    # Type 2: Permutations of (1, 1, ..., 1) with signs, even number of minuses
    
    vectors = []
    
    # Type 1: 2 non-zero entries, values +/-2
    # Choose 2 positions out of 8
    for pos in itertools.combinations(range(8), 2):
        p1, p2 = pos
        # Signs for the two positions
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[p1] = 2 * s1
                v[p2] = 2 * s2
                vectors.append(v)
                
    # Type 2: 8 non-zero entries, values +/-1, even number of -1s
    # There are 2^8 = 256 sign combinations. Half have even number of minuses.
    for signs in itertools.product([1, -1], repeat=8):
        # Check if number of -1s is even
        if sum(1 for s in signs if s == -1) % 2 == 0:
            v = [0] * D
            for i, s in enumerate(signs):
                v[i] = s
            vectors.append(v)
            
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    # Verification (optional, for safety in dev, but kept fast)
    # The logic above guarantees norm_sq=8 and min_dist_sq=8.
    # We return the array.
    
    return result

# EVOLVE-BLOCK-END
