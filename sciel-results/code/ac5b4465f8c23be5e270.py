import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # Type 1: (±2, ±2, 0, ..., 0) in first 10 dims, 0 in 11th
    for i in range(10):
        for j in range(i+1, 10):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
                    
    # Type 2: (±1, ..., ±1) in first 10 dims with even minus count, 0 in 11th
    # Generate all 10-bit vectors, check parity
    for bits in itertools.product([1, -1], repeat=10):
        minus_count = sum(1 for b in bits if b == -1)
        if minus_count % 2 == 0:
            v = list(bits) + [0]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
