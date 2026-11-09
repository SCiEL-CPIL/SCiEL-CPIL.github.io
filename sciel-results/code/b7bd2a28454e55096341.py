import numpy as np
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates using the
    direct sum of the 2*E8 lattice and the 2*D3 lattice.

    The construction yields 252 points:
      - 240 points from the minimal vectors of 2*E8 (embedded in first 8 dimensions)
      - 12 points from the minimal vectors of 2*D3 (embedded in last 3 dimensions)

    All points have squared norm 8, and the minimum squared distance between distinct
    points is 8, satisfying the constraint that min pairwise dist^2 >= max norm^2.

    Returns:
        points: np.ndarray of shape (252, 11) with dtype int64
    """
    d = 11
    points = []

    # Generate minimal vectors of 2*E8
    # Type 1: Two coordinates are ±2, rest are 0
    for i in range(8):
        for j in range(i + 1, 8):
            for sign_i in (1, -1):
                for sign_j in (1, -1):
                    vec = np.zeros(8, dtype=np.int64)
                    vec[i] = 2 * sign_i
                    vec[j] = 2 * sign_j
                    # Embed in 11 dimensions: append three zeros
                    point = np.zeros(11, dtype=np.int64)
                    point[:8] = vec
                    points.append(point)

    # Type 2: All coordinates are ±1 with even number of minus signs
    for mask in range(2 ** 8):
        # Count number of -1s (bits set to 1 in our convention)
        # We'll use bit 1 to mean -1, bit 0 to mean +1
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

    # Generate minimal vectors of 2*D3
    # Permutations of (±2, ±2, 0)
    # Positions of the zero: index 0, 1, or 2 in the last 3 dimensions
    for zero_pos in range(3):
        for sign1 in (1, -1):
            for sign2 in (1, -1):
                vec = np.zeros(3, dtype=np.int64)
                # The two non-zero positions
                non_zero_positions = [i for i in range(3) if i != zero_pos]
                vec[non_zero_positions[0]] = 2 * sign1
                vec[non_zero_positions[1]] = 2 * sign2
                point = np.zeros(11, dtype=np.int64)
                point[8:] = vec
                points.append(point)

    points_array = np.array(points, dtype=np.int64)
    points_array = np.around(points_array).astype(np.int64)

    return points_array
