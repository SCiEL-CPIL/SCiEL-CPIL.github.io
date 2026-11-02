"""Geometric Constructive Strategy: E8 x A3 Lattice Embedding."""
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    D = 11
    vectors = []

    # 1. Generate E8 roots (240 vectors)
    # E8 roots are vectors in Z^8 with norm^2 = 2.
    # They are:
    # a) Weight 2: two +/-1, rest 0. (55 * 4 = 220 vectors)
    # b) Weight 8: all +/-1, even number of minus signs. (128 vectors)
    # Total 240.
    
    # a) Weight 2 in first 8 dims
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * 8
                    v[i] = s1
                    v[j] = s2
                    # Pad to 11 dims
                    v.extend([0, 0, 0])
                    vectors.append(v)
                    
    # b) Weight 8 in first 8 dims
    # All combinations of signs with even number of -1s
    for signs in np.ndindex(2, 2, 2, 2, 2, 2, 2, 2):
        # Map 0/1 to 1/-1
        s = [1 if x == 0 else -1 for x in signs]
        # Check even number of -1s
        if s.count(-1) % 2 == 0:
            v = s + [0, 0, 0]
            vectors.append(v)

    # 2. Generate A3 (or D3) roots in last 3 dims (12 vectors)
    # A3 roots in R3 are weight 2 vectors.
    # (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0)
    # (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1)
    # (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
    
    for i in range(3):
        for j in range(i + 1, 3):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * 8
                    v.extend([0, 0, 0])
                    v[8 + i] = s1
                    v[8 + j] = s2
                    vectors.append(v)

    # Total 240 + 12 = 252 vectors.
    # All have norm^2 = 2.
    # Min dist^2 = 2 (if disjoint support in same block) or 4 (if different blocks).
    # Wait, E8 roots have min dist^2 = 2.
    # A3 roots have min dist^2 = 2.
    # Cross dist^2 = 2 + 2 = 4.
    # So min dist^2 = 2.
    # Max norm^2 = 2.
    # Condition: min dist >= max norm.
    # sqrt(2) >= sqrt(2). True.
    
    result = np.array(vectors, dtype=np.int64)
    
    # Optional: Try to add more?
    # The directive says bypass graph optimization.
    # So we return this deterministic set.
    
    return result
