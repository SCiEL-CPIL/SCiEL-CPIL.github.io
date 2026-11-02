"""Construct kissing configuration in R^11 using W4 vectors (two non-zero entries of magnitude 1)."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    start_time = time.time()
    
    # Generate all possible W4 vectors: vectors with exactly two non-zero entries, each ±1.
    # There are C(11, 2) pairs and 4 sign combinations = 55 * 4 = 220 vectors.
    # All these vectors have norm sqrt(2).
    # The minimum squared distance between any two distinct W4 vectors is 2.
    # (Case: share 1 coord, same sign -> dist^2 = 2. Case: share 0 coords -> dist^2 = 4. Case: share 1 coord, opp sign -> dist^2 = 6.)
    # The maximum norm is sqrt(2).
    # The condition "minimum pairwise distance >= maximum norm" becomes sqrt(2) >= sqrt(2), which is true.
    # Thus, the set of ALL 220 W4 vectors is a valid kissing configuration.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
                    
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
