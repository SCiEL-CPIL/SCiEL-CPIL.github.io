"""
Rewritten program for Kissing Number in Dimension 11.
Uses the lattice E8 (scaled to integer vectors) in the first 8 dimensions
and D3 (scaled to integer vectors) in the last 3 dimensions.
Total vectors: 240 (from E8) + 12 (from D3) = 252.
"""

import numpy as np

def construct():
    D = 11
    vectors = []
    
    # --- Generate E8 minimal vectors (scaled by 2) in first 8 dims ---
    # Norm squared will be 8.
    
    # Type 1: 112 vectors of form (+/- 2, +/- 2, 0, ...)
    # Indices 0..7
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = 2 * si
                    v[j] = 2 * sj
                    vectors.append(v)
    
    # Type 2: 128 vectors of form (+/- 1, ..., +/- 1) with even number of minus signs
    # We generate all 256 sign combinations and filter for even minus signs
    # Note: The standard E8 definition uses even number of minus signs for the half-integer part.
    # Let's verify: (1/2)(1,1,1,1,1,1,1,1) has 0 minuses (even).
    # (1/2)(-1,-1,-1,-1,1,1,1,1) has 4 minuses (even).
    # So we select combinations with even number of -1s.
    
    for signs in np.ndindex(2**8):
        # Convert binary index to signs
        # 0 -> 1, 1 -> -1
        s = [1 if (signs[0] >> i) & 1 == 0 else -1 for i in range(8)]
        
        # Count minus signs
        minus_count = sum(1 for x in s if x == -1)
        
        if minus_count % 2 == 0:
            v = [0] * D
            for k in range(8):
                v[k] = s[k]
            vectors.append(v)
            
    # --- Generate D3 minimal vectors (scaled by 2) in last 3 dims ---
    # Indices 8, 9, 10
    # Pairs: (8,9), (8,10), (9,10)
    for i, j in [(8, 9), (8, 10), (9, 10)]:
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = 2 * si
                v[j] = 2 * sj
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int64)
