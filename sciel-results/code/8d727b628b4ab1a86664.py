import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    The set consists of:
    1. 240 points from the 8-dimensional component (norm squared 8).
    2. 12 points from the 3-dimensional component (norm squared 8).
    3. 96 mixed points with one coordinate ±2 in the first 8 dims and one coordinate ±2 in the last 3 dims (norm squared 8).
    
    Total points: 252 + 96 = 348.
    All points have squared norm 8.
    Minimum squared distance is 8.
    """
    d = 11
    points = []

    # Generate minimal vectors of 2*E8 (actually norm 8 vectors in 8D)
    # Type 1: Two coordinates are ±2, rest are 0
    for i in range(8):
        for j in range(i + 1, 8):
            for sign_i in (1, -1):
                for sign_j in (1, -1):
                    vec = np.zeros(8, dtype=np.int64)
                    vec[i] = 2 * sign_i
                    vec[j] = 2 * sign_j
                    point = np.zeros(11, dtype=np.int64)
                    point[:8] = vec
                    points.append(point)

    # Type 2: All coordinates are ±1 with even number of minus signs
    for mask in range(2 ** 8):
        minus_count = bin(mask).count('1')
        if minus_count % 2 == 0:
            vec = np.zeros(8, dtype=np.int64)
            for i in range(8):
                if (mask >> i) & 1:
                    vec[i] = -1
                else:
                    vec[i] = 1
            point = np.zeros(11, dtype=np.int64)
            point[:8] = vec
            points.append(point)

    # Generate minimal vectors of 2*D3 (norm 8 vectors in 3D)
    for zero_pos in range(3):
        for sign1 in (1, -1):
            for sign2 in (1, -1):
                vec = np.zeros(3, dtype=np.int64)
                non_zero_positions = [i for i in range(3) if i != zero_pos]
                vec[non_zero_positions[0]] = 2 * sign1
                vec[non_zero_positions[1]] = 2 * sign2
                point = np.zeros(11, dtype=np.int64)
                point[8:] = vec
                points.append(point)
                
    # Generate mixed vectors: one ±2 in first 8, one ±2 in last 3
    for i in range(8):
        for j in range(3):
            for sign_i in (1, -1):
                for sign_j in (1, -1):
                    point = np.zeros(11, dtype=np.int64)
                    point[i] = 2 * sign_i
                    point[8 + j] = 2 * sign_j
                    points.append(point)

    points_array = np.array(points, dtype=np.int64)
    points_array = np.around(points_array).astype(np.int64)

    return points_array
