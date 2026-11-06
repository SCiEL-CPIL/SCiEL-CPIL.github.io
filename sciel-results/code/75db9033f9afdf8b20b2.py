"""
Strategy: Pseudo-random permutation priority via high-quality hashing.

Instead of trying to find a "smart" static ordering (like sum-based or linear projections)
which often leads to local clustering and suboptimal greedy choices, we use a
cryptographically strong hash function to assign a pseudo-random priority to each vector.

The intuition is that a "random" ordering of the hypercube {0,1,2}^n is a robust
heuristic for greedy set construction problems like cap sets. It avoids the
systematic biases of simple arithmetic priorities. By using a good hash (like a
modified version of xxHash or a simple but effective mixing function), we ensure
that the ordering is effectively a random permutation, which tends to explore the
space more uniformly and avoids getting stuck in bad local structures early on.

This is a structural change from "searching for the best heuristic" to "using a
robust, non-adaptive randomization strategy".
"""

import time
import math
import itertools
import random

# EVOLVE-BLOCK-START

# We don't need complex offline search. We just need a good hash function.
# The priority function itself will be the hash.

def _mix64(x):
    """
    A simple but effective 64-bit mixing function.
    Based on MurmurHash3 finalizer or similar.
    """
    x = (x ^ (x >> 33)) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 33)) & 0xFFFFFFFFFFFFFFFF
    x = (x * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 33)) & 0xFFFFFFFFFFFFFFFF
    return x

def _vector_to_int(v, n):
    """
    Convert a vector in {0,1,2}^n to a unique integer in [0, 3^n).
    This is just the base-3 representation.
    """
    val = 0
    for i in range(n):
        val = val * 3 + v[i]
    return val

def priority(vector, n):
    """
    Returns a pseudo-random priority for the vector.
    The goal is to create a random permutation of the vectors.
    """
    # Convert vector to a unique integer ID
    vid = _vector_to_int(vector, n)
    
    # Mix the ID to get a pseudo-random 64-bit integer
    # We add a fixed seed to ensure determinism across runs
    h = _mix64(vid + 123456789)
    
    # Return as a float. Since the hash is uniform in [0, 2^64),
    # dividing by 2^64 gives a uniform float in [0, 1).
    # However, for sorting, we can just return the float value directly.
    # The greedy algorithm sorts by descending priority.
    return float(h) / (1 << 64)

# EVOLVE-BLOCK-END
