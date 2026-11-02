"""Program to find a large kissing configuration in R^11 using integer vectors."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Generate all weight-2 vectors: for each pair (i,j) with i<j, and signs (si,sj)
    # Total: C(11,2) * 4 = 55 * 4 = 220
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    
    # Verify the constraint: min pairwise distance >= max norm
    # All vectors have norm sqrt(2)
    # Pairwise distance between two weight-2 vectors:
    # - If disjoint supports: dist^2 = 4, dist = 2 > sqrt(2)
    # - If share 1 index: dist^2 = 2 or 4 depending on signs, min is 2, dist = sqrt(2) >= sqrt(2)
    # So the condition is satisfied.
    
    return vectors
# EVOLVE-BLOCK-END
