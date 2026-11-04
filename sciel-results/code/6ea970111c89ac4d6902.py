import numpy as np
import itertools

def construct():
    D = 11
    vectors = []
    
    # 1. Generate E8 minimal vectors (scaled by 2)
    # Part A: Integer part of E8 (two non-zero +/-1 in E8 -> +/-2 in scaled)
    # Vectors in Z^8 with two non-zero entries +/-1 and even sum (always even for 2 non-zeros: 1+1=2, 1-1=0)
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    # Part B: Half-integer part of E8 (four non-zero +/-1/2 in E8 -> +/-1 in scaled)
    # Vectors in (Z+1/2)^8 with all coords +/-1/2 and even number of +1/2s
    # Scaled: all coords +/-1, even number of +1s
    # We need to choose positions for non-zeros? No, in the half-integer shell of E8, 
    # all 8 coordinates are non-zero.
    # Wait, are there minimal vectors in E8 with fewer than 4 non-zero half-integers?
    # Norm^2 = sum x_i^2 = 2.
    # If x_i = +/- 1/2, then number of non-zeros must be 8 * (1/4) = 2? No.
    # 8 * (1/4) = 2. So yes, all 8 coordinates are non-zero.
    # So we generate all vectors in {+1, -1}^8 with an even number of +1s.
    for mask in range(256):
        # Count bits to determine number of +1s (if we map 1 to +1)
        # Let's just iterate all 2^8 combinations
        pass
    
    # Better: iterate all 2^8 vectors of +/-1
    for bits in range(256):
        v = [0] * 11
        num_plus = 0
        for i in range(8):
            if (bits >> i) & 1:
                v[i] = 1
                num_plus += 1
            else:
                v[i] = -1
        # Condition: even number of +1s
        if num_plus % 2 == 0:
            vectors.append(v)
            
    # 2. Generate D3 minimal vectors (scaled by 2)
    # D3 roots: two non-zero +/-1 in 3D.
    # Scaled: two non-zero +/-2 in 3D.
    # Embedded in R^11 at indices 8, 9, 10.
    for i, j in itertools.combinations(range(8, 11), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * 11
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
