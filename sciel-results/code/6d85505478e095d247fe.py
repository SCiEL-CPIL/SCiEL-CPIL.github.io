# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    such that their minimum pairwise squared distance is >= their maximum squared norm.
    Aims to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    deadline = time.time() + 580  # safety margin before 600s limit

    # Generate candidate vectors
    # We'll use vectors with entries in {-1, 0, 1}
    # Norm^2 = number of non-zero entries
    # For the constraint: min pairwise dist^2 >= max norm^2
    # If all points have norm^2 = r, then we need min pairwise dist^2 >= r
    # For vectors with entries in {-1,0,1}, dist^2 = 4 - 4*dot (if both have same support pattern)
    # More generally, for u, v with entries in {-1,0,1}:
    #   ||u-v||^2 = ||u||^2 + ||v||^2 - 2*u.v
    # If ||u||^2 = ||v||^2 = r, then ||u-v||^2 = 2r - 2*u.v
    # We need 2r - 2*u.v >= r, i.e., r >= 2*u.v, i.e., u.v <= r/2

    # Strategy: use vectors with norm^2 = 2 (exactly 2 non-zero entries)
    # Then we need u.v <= 1 for all pairs
    # Two such vectors: if they share one coordinate, u.v = ±1 (ok if sign makes it 1, bad if -1... wait)
    # Actually u.v can be -1, 0, or 1 when both have norm^2=2
    # We need u.v <= 1, which is always true since max u.v = 1 (when they're identical)
    # But we also need distinct points, so u != v, meaning u.v < 2 (always true for norm^2=2)
    # Wait, we need dist^2 >= max norm^2 = 2
    # dist^2 = 2*2 - 2*u.v = 4 - 2*u.v >= 2 means u.v <= 1
    # For norm^2=2 vectors, u.v is in {-2,-1,0,1,2} but since they're distinct, u.v != 2
    # u.v = 1: dist^2 = 4-2 = 2 >= 2 ✓
    # u.v = 0: dist^2 = 4 >= 2 ✓
    # u.v = -1: dist^2 = 4+2 = 6 >= 2 ✓
    # u.v = -2: only if v = -u, dist^2 = 8 >= 2 ✓
    # So ANY two distinct norm^2=2 vectors satisfy the constraint!
    
    # How many norm^2=2 vectors are there?
    # Choose 2 positions out of 11: C(11,2) = 55
    # Each position can be -1 or 1: 4 sign combinations
    # Total: 55 * 4 = 220
    # But we need to check: are all pairs valid? Yes, as shown above.
    # So we can use ALL 220 vectors!
    
    # Wait, let me double-check. If u and v are both norm^2=2 and distinct:
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 - 2<u,v>
    # <u,v> max is 1 (when they share one coordinate with same sign)
    # <u,v> min is -1 (when they share one coordinate with opposite sign, or are negatives... 
    #   actually if u = (1,1,0,...) and v = (-1,-1,0,...) then <u,v> = -2, but v = -u)
    # For distinct u,v with norm^2=2:
    #   If they share no coordinates: <u,v> = 0
    #   If they share 1 coordinate: <u,v> = ±1
    #   If they share 2 coordinates: u = ±v, so <u,v> = ±2
    # So <u,v> ∈ {-2,-1,0,1} for distinct u,v (excluding <u,v>=2 which means u=v)
    # dist^2 = 4 - 2<u,v> ∈ {4-2(-2), 4-2(-1), 4-2(0), 4-2(1)} = {8, 6, 4, 2}
    # All >= 2 = max norm^2 ✓
    
    # So we can use all 220 vectors with exactly 2 non-zero entries!
    
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # All 220 are mutually compatible
    points = np.array(candidates, dtype=np.int64)
    
    # Verify the constraint
    k = len(points)
    max_norm_sq = 0
    for p in points:
        n2 = np.dot(p, p)
        if n2 > max_norm_sq:
            max_norm_sq = n2
    
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            d2 = np.dot(diff, diff)
            if d2 < min_dist_sq:
                min_dist_sq = d2
    
    # Check validity
    if min_dist_sq >= max_norm_sq and k >= 2:
        return np.around(points).astype(np.int64)
    
    # Fallback: return the 2 points from seed
    points = np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    return np.around(points).astype(np.int64)


# EVOLVE-BLOCK-END
