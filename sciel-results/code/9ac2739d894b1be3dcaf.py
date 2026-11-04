import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    D = 11
    
    # E8 lattice kissing vectors (scaled to integers)
    # Type 1: Permutations of (+-2, +-2, 0, 0, 0, 0, 0, 0)
    # Type 2: Permutations of (+-1, +-1, +-1, +-1, +-1, +-1, +-1, +-1) with even number of -1s
    
    vectors = []
    
    # Generate Type 1
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    v = np.zeros(8, dtype=np.int8)
                    v[i] = 2 * s1
                    v[j] = 2 * s2
                    vectors.append(v)
    
    # Generate Type 2
    # Iterate over all 2^8 sign combinations
    for mask in range(256):
        # Count minuses
        num_minuses = 0
        for k in range(8):
            if (mask >> k) & 1:
                num_minuses += 1
        if num_minuses % 2 == 0:
            v = np.zeros(8, dtype=np.int8)
            for k in range(8):
                if (mask >> k) & 1:
                    v[k] = -1
                else:
                    v[k] = 1
            vectors.append(v)
    
    # Pad to dimension 11
    if vectors:
        vec_array = np.array(vectors, dtype=np.int8)
        # Pad with 0s on the right
        padded = np.zeros((vec_array.shape[0], D), dtype=np.int8)
        padded[:, :8] = vec_array
        return padded
    else:
        # Fallback if something went wrong (should not happen)
        return np.zeros((0, D), dtype=np.int8)
