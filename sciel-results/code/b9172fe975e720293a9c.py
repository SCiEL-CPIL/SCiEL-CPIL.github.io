# EVOLVE-BLOCK-START
import numpy as np
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer lattice points with the property that
    the minimum squared pairwise distance is >= the maximum squared norm.
    
    Uses all points with exactly 2 nonzero coordinates (each ±1), giving C(11,2)*4 = 220 points.
    All have norm^2 = 2, and minimum pairwise distance^2 = 3 >= 2.
    """
    deadline = time.time() + 600.0
    
    d = 11
    points_list = []
    
    # Generate all points with exactly 2 nonzero coordinates, each being +1 or -1
    for i in range(d):
        for j in range(i + 1, d):
            for si in (-1, 1):
                for sj in (-1, 1):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify constraints
    k = len(points)
    assert k >= 2
    assert points.shape[1] == 11
    
    # No zero vector
    for p in points:
        assert np.any(p != 0), "Zero vector found"
    
    # Check min pairwise distance^2 >= max norm^2
    max_norm_sq = 0
    for p in points:
        ns = int(np.dot(p, p))
        if ns > max_norm_sq:
            max_norm_sq = ns
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            ds = int(np.dot(diff, diff))
            if ds < min_dist_sq:
                min_dist_sq = ds
    
    assert min_dist_sq >= max_norm_sq, f"min_dist_sq={min_dist_sq} < max_norm_sq={max_norm_sq}"
    
    # Round and cast as required
    points = np.around(points).astype(np.int64)
    
    return points


# EVOLVE-BLOCK-END
