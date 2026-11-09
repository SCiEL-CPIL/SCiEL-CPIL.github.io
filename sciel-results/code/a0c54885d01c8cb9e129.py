import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    points = []
    
    # Type 1: 112 vectors from Z^8 part of E8
    # Two non-zero coordinates, each +/- 1
    for i in range(8):
        for j in range(i + 1, 8):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(8, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
                    
    # Type 2: 128 vectors from (Z+1/2)^8 part of E8
    # Eight non-zero coordinates, each +/- 1/2
    # Must have even number of +1/2
    for mask in range(256):
        # Count bits
        if bin(mask).count('1') % 2 == 0:
            v = np.zeros(8, dtype=np.int64)
            for k in range(8):
                if mask & (1 << k):
                    v[k] = 1
                else:
                    v[k] = -1
            # Scale by 2 to make integers (1/2 -> 1)
            # Wait, if we use 1/2, we need to scale the whole set by 2.
            # If we generate 1/2s, we can't store in int64 easily without scaling.
            # So generate the 1/2 version as integers by multiplying by 2 immediately?
            # No, let's just generate the scaled integers directly.
            # Type 2 scaled: coordinates are +/- 1.
            # Condition: even number of +1s.
            # The logic above generates +/- 1 directly.
            points.append(v)
            
    # Scale Type 1 by 2 as well to match norms
    # Type 1 was +/- 1, +/- 1. Norm 2.
    # Type 2 is +/- 1 ... +/- 1. Norm 8.
    # We need them to have the same norm for the kissing property to hold uniformly?
    # Actually, the E8 lattice has all minimal vectors at norm 2.
    # Type 1 (in Z^8) has norm 2.
    # Type 2 (in (Z+1/2)^8) has norm 8 * (1/4) = 2.
    # So they have the same norm.
    # When we scale by 2:
    # Type 1 becomes norm 8.
    # Type 2 becomes norm 8.
    # So we must scale Type 1 by 2.
    
    scaled_points = []
    for p in points[:112]:
        scaled_points.append(p * 2)
    for p in points[112:]:
        scaled_points.append(p) # Already effectively scaled?
        # Wait.
        # In the loop above for Type 2, I set v[k] = 1 or -1.
        # This corresponds to 2 * (1/2) = 1.
        # So Type 2 is already the scaled version.
        # Type 1 was 1, -1. Scaled is 2, -2.
        # So yes, scale first 112 by 2.
        
    # Combine
    all_points = np.array(scaled_points, dtype=np.int64)
    
    # Pad to 11 dimensions
    padded = np.zeros((len(all_points), 11), dtype=np.int64)
    padded[:, :8] = all_points
    
    return padded
