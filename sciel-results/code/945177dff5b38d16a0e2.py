import numpy as np
import itertools
import time

D = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # 100s limit to be safe
    
    vectors = []
    
    # 1. Generate E8 minimal vectors (embedded in first 8 dims)
    # Type 1: Two non-zero coordinates +/-1
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0.0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    # Type 2: Eight non-zero coordinates +/-0.5, even number of minuses
    # Generate all 2^7 = 128 combinations
    # We can iterate over the first 7 bits, the 8th is determined by parity
    for bits in range(128):
        v = [0.0] * D
        num_neg = 0
        for k in range(7):
            if (bits >> k) & 1:
                v[k] = -0.5
                num_neg += 1
            else:
                v[k] = 0.5
        # The 8th coordinate (index 7)
        # We need total number of minuses to be even.
        if num_neg % 2 == 0:
            v[7] = -0.5
        else:
            v[7] = 0.5
        vectors.append(v)
        
    # 2. Generate A3 minimal vectors (embedded in last 3 dims, indices 8,9,10)
    # A3 roots are vectors with two non-zero coords +/-1 in 3D
    # Coordinates are in dims 8, 9, 10
    for i, j in itertools.combinations([8, 9, 10], 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0.0] * D
                v[i] = si
                v[j] = sj
                vectors.append(v)
                
    # Total vectors should be 112 + 128 + 12 = 252
    # Convert to integer by multiplying by 2
    # This makes norms 2 and min distance 2
    # Condition: min dist >= max norm -> 2 >= 2 (True)
    
    vec_array = np.array(vectors, dtype=np.float64)
    vec_array = vec_array * 2
    
    # Round to integer to avoid floating point issues
    vec_int = np.round(vec_array).astype(np.int64)
    
    # Verify validity (optional, but good for safety)
    # If time permits, check min dist and max norm
    if time.time() < deadline:
        # Calculate norms
        norms = np.linalg.norm(vec_int, axis=1)
        max_norm = np.max(norms)
        
        # Calculate min distance
        # 252 vectors, 252x252 distance matrix is small
        # dist^2 = norm_u^2 + norm_v^2 - 2 u.v
        # All norms are 2, so norm^2 = 4
        # dist^2 = 8 - 2 u.v
        # We need dist >= 2 -> dist^2 >= 4 -> 8 - 2 u.v >= 4 -> u.v <= 2
        # Since u, v are integer vectors with norm 2, max dot product is 2 (if u=v)
        # For u != v, max dot product is 1?
        # If u=(2,0..), v=(1,1,0..) not in set.
        # In E8 min vectors, max dot product for distinct vectors is 1.
        # So dist^2 = 8 - 2(1) = 6? No.
        # Let's check u=(1,1,0..), v=(1,0,1..). Dot=1.
        # Scaled by 2: u=(2,2,0..), v=(2,0,2..). Dot=4.
        # Norm^2 = 8.
        # dist^2 = 8 + 8 - 2(4) = 8. dist = 2.82.
        # Wait, min dist in E8 min vectors is sqrt(2).
        # Scaled by 2, min dist is 2*sqrt(2) = 2.82.
        # Max norm is 2*sqrt(2) = 2.82.
        # Condition: 2.82 >= 2.82. Holds.
        
        pass
        
    return vec_int

# EVOLVE-BLOCK-END
