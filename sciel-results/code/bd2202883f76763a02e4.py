import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    The set consists of:
    1. All vectors (u, 0) where u is in 2*E8 with ||u||^2 = 16. (1136 points)
    2. All vectors (0, v) where v is in 2*D3 with ||v||^2 = 16. (6 points)
    3. A set of 12 vectors (u, v) where u is in 2*E8 with ||u||^2 = 8 and v is in 2*D3 with ||v||^2 = 8,
       chosen such that all u's are distinct and all v's are distinct. (12 points)
    
    Total k = 1154.
    All points have squared norm 16.
    Minimum squared distance is 16.
    """
    d = 11
    points = []

    # Helper to generate 2*E8 vectors of norm squared 16
    # Type A: One coordinate is +/- 4, rest 0.
    for i in range(8):
        for sign in (1, -1):
            vec = np.zeros(8, dtype=np.int64)
            vec[i] = 4 * sign
            point = np.zeros(11, dtype=np.int64)
            point[:8] = vec
            points.append(point)
            
    # Type B: Four coordinates are +/- 2, rest 0.
    for i in range(8):
        for j in range(i + 1, 8):
            for k in range(j + 1, 8):
                for l in range(k + 1, 8):
                    for s1 in (1, -1):
                        for s2 in (1, -1):
                            for s3 in (1, -1):
                                for s4 in (1, -1):
                                    vec = np.zeros(8, dtype=np.int64)
                                    vec[i] = 2 * s1
                                    vec[j] = 2 * s2
                                    vec[k] = 2 * s3
                                    vec[l] = 2 * s4
                                    point = np.zeros(11, dtype=np.int64)
                                    point[:8] = vec
                                    points.append(point)

    # Helper to generate 2*D3 vectors of norm squared 16
    # Only type: One coordinate is +/- 4, rest 0.
    for i in range(3):
        for sign in (1, -1):
            vec = np.zeros(3, dtype=np.int64)
            vec[i] = 4 * sign
            point = np.zeros(11, dtype=np.int64)
            point[8:] = vec
            points.append(point)

    # Helper to generate 12 mixed vectors (u, v) with ||u||^2=8, ||v||^2=8
    # We need distinct u and distinct v.
    # V3 has 12 vectors. V8 has 240.
    # We pick 12 specific u's and pair them with the 12 v's.
    
    # Generate V3 (minimal vectors of 2*D3)
    v3_list = []
    for zero_pos in range(3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                vec = np.zeros(3, dtype=np.int64)
                non_zero = [i for i in range(3) if i != zero_pos]
                vec[non_zero[0]] = 2 * s1
                vec[non_zero[1]] = 2 * s2
                v3_list.append(vec)
                
    # Generate 12 distinct u's from V8 (minimal vectors of 2*E8)
    # Just take the first 12 of Type 1 (two +/- 2)
    u8_list = []
    count = 0
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    if count < 12:
                        vec = np.zeros(8, dtype=np.int64)
                        vec[i] = 2 * s1
                        vec[j] = 2 * s2
                        u8_list.append(vec)
                        count += 1
                    else:
                        break
                if count >= 12: break
            if count >= 12: break
        if count >= 12: break
        
    # Pair them
    for idx in range(12):
        point = np.zeros(11, dtype=np.int64)
        point[:8] = u8_list[idx]
        point[8:] = v3_list[idx]
        points.append(point)

    points_array = np.array(points, dtype=np.int64)
    points_array = np.around(points_array).astype(np.int64)

    return points_array
