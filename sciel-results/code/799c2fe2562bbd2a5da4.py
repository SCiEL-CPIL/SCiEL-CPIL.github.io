import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    
    # Generate all weight-2 vectors
    # A weight-2 vector has exactly two non-zero entries, each +1 or -1.
    # Indices: (i, j) with i < j.
    # Signs: (1, 1), (1, -1), (-1, 1), (-1, -1).
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors.append(v)
                    
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    # Verification (optional, but good for safety)
    # Norm squared is 2. Norm is sqrt(2).
    # Min pairwise distance squared must be >= 2.
    # For distinct weight-2 vectors, min dist squared is 2.
    
    return result
