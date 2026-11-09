"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # Generate all nonzero integer vectors of norm squared 2 (weight 2)
    # These are vectors with exactly two non-zero entries, each being +1 or -1.
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    result = np.array(vectors, dtype=np.int64)
    
    # Verify validity just in case
    # R = sqrt(2)
    # Min dist sq must be >= 2
    # We know this is true for this set.
    
    return result

# EVOLVE-BLOCK-END
