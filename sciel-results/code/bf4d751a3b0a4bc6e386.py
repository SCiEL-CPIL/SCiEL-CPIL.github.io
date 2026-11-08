"""Construction of a kissing configuration in R^11 using integer vectors."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110  # safety margin from 150s limit

    # We want nonzero integer vectors v in Z^11 such that:
    #   min_{i != j} ||v_i - v_j|| >= max_i ||v_i||
    #
    # Strategy: use all weight-2 vectors (exactly two nonzero entries, each ±1).
    # Each has norm squared = 2, so max norm = sqrt(2).
    # For two distinct weight-2 vectors, the dot product is in {-2, -1, 0, 1}.
    # (It equals 2 only if they are identical, which we exclude.)
    # Distance squared = 2 + 2 - 2*(dot) = 4 - 2*dot >= 4 - 2*1 = 2.
    # So min distance squared >= 2 = max norm squared. Condition satisfied.
    #
    # Total count: C(11,2) * 4 = 55 * 4 = 220.
    #
    # Could we do better with a mixed set?
    # If we include any vector with norm > sqrt(2), the max norm increases.
    # For example, weight-3 vectors have norm sqrt(3). Then we'd need
    # min distance >= sqrt(3), i.e., distance squared >= 3.
    # This is a much stricter constraint and likely reduces the total count.
    #
    # The weight-2 construction gives 220 vectors, which is the kissing number
    # of the D_11 lattice (a well-known result in sphere packing theory).
    #
    # Generate all weight-2 integer vectors in R^11.
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

    if time.time() > deadline:
        return result

    return result
# EVOLVE-BLOCK-END
