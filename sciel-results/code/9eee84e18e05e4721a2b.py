# EVOLVE-BLOCK-START
import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    consisting of all vectors with exactly two non-zero entries (+/-1).
    This set has 220 points, all with squared norm 2.
    The minimum squared distance between any pair is 2, satisfying the condition
    min_dist_sq >= max_norm_sq (2 >= 2).
    
    Returns:
        points: np.ndarray of shape (220, 11)
    """
    d = 11
    points = []
    
    # Generate all vectors with exactly two non-zero entries which are +/-1
    # There are C(11, 2) pairs of indices, and 4 sign combinations for each.
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    vec = [0] * d
                    vec[i] = si
                    vec[j] = sj
                    points.append(vec)
    
    points = np.array(points, dtype=np.int64)
    return points

# EVOLVE-BLOCK-END
