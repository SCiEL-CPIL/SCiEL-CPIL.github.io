"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct():
    # We use weight-7 vectors.
    # Norm squared is 7. Max norm is sqrt(7).
    # Condition for compatibility: ||u - v||^2 >= 7
    # ||u||^2 + ||v||^2 - 2 u.v >= 7
    # 7 + 7 - 2 u.v >= 7 => 7 - 2 u.v >= 0 => u.v <= 3.5 => u.v <= 3.
    # For u, v in same support S (size 7), u.v = 7 - 2*h(u,v).
    # 7 - 2*h <= 3 => 4 <= 2*h => h >= 2.
    # We need a set of binary vectors of length 7 with min Hamming distance 2.
    # Max size is 2^(7-1) = 64.
    # Construction: Fix the first bit to 0.
    
    vectors = []
    # Generate all subsets of indices of size 7
    for support in itertools.combinations(range(D), 7):
        # For this support, generate all 2^7 sign combinations
        # But only those where the sign at support[0] is +1 (index 0 in the local tuple)
        # We can iterate over the remaining 6 bits
        for bits in itertools.product([1, -1], repeat=6):
            v = [0] * D
            # Set the first bit to +1
            v[support[0]] = 1
            # Set the remaining 6 bits
            for i in range(6):
                v[support[i+1]] = bits[i]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
