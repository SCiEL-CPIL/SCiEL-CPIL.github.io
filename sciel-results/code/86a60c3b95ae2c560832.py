# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Strategy: Use the E8 ⊕ A3 lattice construction.
    - E8 roots: 240 vectors in 8D with squared norm 2.
    - A3 roots: 12 vectors in 3D with squared norm 2.
    - Points are formed by:
      1. E8 roots embedded in first 8 dims (last 3 are 0): 240 points, norm^2 = 2.
      2. A3 roots embedded in last 3 dims (first 8 are 0): 12 points, norm^2 = 2.
      3. Combinations? No, we need distinct points with min pair dist >= max norm.
      
    Let's analyze the distance conditions:
    All points have norm^2 = 2. So max norm^2 = 2.
    We need min pair dist^2 >= 2.
    
    For two E8 roots u, v (embedded in first 8 dims):
    dist^2 = ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 - 2 u.v.
    For E8 roots, u.v can be -1, 0, 1.
    If u.v = 1, dist^2 = 2. (These are adjacent roots)
    If u.v = 0, dist^2 = 4.
    If u.v = -1, dist^2 = 6.
    So min dist^2 among E8 roots is 2. This satisfies dist^2 >= 2.
    
    For two A3 roots a, b (embedded in last 3 dims):
    dist^2 = 4 - 2 a.b.
    For A3 roots, a.b can be -1, 0, 1.
    If a.b = 1, dist^2 = 2.
    So min dist^2 among A3 roots is 2. This satisfies dist^2 >= 2.
    
    For an E8 root u and an A3 root a:
    u = (u1, ..., u8, 0, 0, 0)
    a = (0, ..., 0, a1, a2, a3)
    u.a = 0.
    dist^2 = ||u||^2 + ||a||^2 - 2(0) = 2 + 2 = 4.
    So dist^2 = 4 >= 2.
    
    Thus, the union of 240 E8 roots and 12 A3 roots gives 252 points, all with norm^2 = 2, and min pair dist^2 = 2.
    This is valid since min dist^2 (2) >= max norm^2 (2).
    
    Wait, the condition is "minimum squared distance over all pairs of points is >= the maximum squared norm of the points".
    Max norm^2 is 2. Min dist^2 is 2. So 2 >= 2 holds.
    
    Total points: 240 + 12 = 252.
    """
    d = 11
    deadline = time.time() + 590.0  # Safety margin
    
    # Generate E8 roots
    # E8 roots are:
    # 1. (±1, ±1, 0, ..., 0) with two non-zero entries: 8*7*2*2 = 224? No.
    #    Number of ways to choose 2 positions from 8: C(8,2) = 28.
    #    For each pair, signs can be (1,1), (1,-1), (-1,1), (-1,-1). That's 4 sign patterns.
    #    But (1,1) and (-1,-1) are negatives, etc. All are distinct roots.
    #    28 * 4 = 112 roots of type (±1, ±1, 0,...)
    # 2. (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs.
    #    Total 2^8 = 256 sign patterns. Half have even number of minuses: 128 roots.
    #    But these have coordinates ±1/2, not integers!
    #    Wait, the problem requires integer coordinates?
    #    "Entries are rounded with np.around and cast to int64; all checks use these integers with exact arithmetic."
    #    "Raw value: k, the number of points."
    #    "Valid only if: ... no point is the zero vector; and the minimum squared distance over all pairs of points is >= the maximum squared norm of the points"
    #    
    #    The task says "Integer kissing configuration". The return type is int64.
    #    So we CANNOT use the 128 half-integer roots of E8.
    #    We can only use the 112 integer roots of E8: (±1, ±1, 0, ..., 0).
    
    # Let's re-evaluate with only integer roots.
    # E8 integer roots: 112 vectors of form (±1, ±1, 0, ..., 0).
    # A3 roots: 12 vectors.
    #   A3 roots are (±1, ±1, 0), (0, ±1, ±1), (±1, 0, ±1).
    #   3 pairs * 4 sign patterns = 12 roots. All integer.
    
    # So we have 112 + 12 = 124 points.
    # Let's check if we can do better.
    # What about other integer vectors of norm^2 = 2?
    # In 11D, integer vectors of norm^2 = 2 are exactly (±1, ±1, 0, ..., 0).
    # Number of such vectors: C(11, 2) * 4 = 55 * 4 = 220.
    # These are all the integer roots of norm^2 = 2 in 11D.
    
    # Let's check the distance between any two such vectors.
    # v = (±1, ±1, 0, ..., 0)
    # w = (±1, ±1, 0, ..., 0)
    # If v and w share no indices: v.w = 0, dist^2 = 4.
    # If v and w share one index: v.w = ±1, dist^2 = 2 or 6.
    # If v and w share two indices: they are identical (dist^2=0) or negatives (dist^2=8) or one is (1,1) and other (1,-1) etc.
    #   If they are distinct and share two indices, they must be of the form:
    #   v = (1, 1, 0, ...) and w = (1, -1, 0, ...). Then v.w = -1. dist^2 = 6.
    #   v = (1, 1, 0, ...) and w = (-1, 1, 0, ...). Then v.w = 0. dist^2 = 4.
    #   v = (1, 1, 0, ...) and w = (-1, -1, 0, ...). Then v.w = -2. dist^2 = 8.
    #   So for distinct vectors sharing two indices, min dist^2 is 4?
    #   Wait, let's list all sign patterns for two indices i, j:
    #   (1,1), (1,-1), (-1,1), (-1,-1).
    #   Pairs of distinct vectors:
    #   (1,1) and (1,-1): dot = 1*1 + 1*(-1) = 0. dist^2 = 4.
    #   (1,1) and (-1,1): dot = 1*(-1) + 1*1 = 0. dist^2 = 4.
    #   (1,1) and (-1,-1): dot = -1-1 = -2. dist^2 = 8.
    #   (1,-1) and (-1,1): dot = -1-1 = -2. dist^2 = 8.
    #   (1,-1) and (-1,-1): dot = -1+1 = 0. dist^2 = 4.
    #   (-1,1) and (-1,-1): dot = 1-1 = 0. dist^2 = 4.
    #   So min dist^2 among distinct vectors sharing two indices is 4.
    
    # What about vectors sharing one index?
    # v = (1, 1, 0, ...), w = (1, 0, 1, 0, ...) (indices 1,2 vs 1,3)
    # v.w = 1*1 + 1*0 + 0*1 = 1. dist^2 = 4 - 2(1) = 2.
    # So min dist^2 is 2.
    
    # So the set of all 220 integer vectors of norm^2 = 2 in 11D has:
    # - Max norm^2 = 2.
    # - Min pair dist^2 = 2.
    # - Condition: 2 >= 2. Valid.
    
    # Can we add more points?
    # We need points with norm^2 <= 2 (since max norm^2 is currently 2, and we need min dist^2 >= max norm^2).
    # If we add a point with norm^2 = 1, then max norm^2 becomes 2 (if we keep the norm^2=2 points) or 1 (if we remove them).
    # If max norm^2 is 1, we need min dist^2 >= 1.
    # Integer vectors of norm^2 = 1 are (±1, 0, ..., 0). There are 22 such vectors.
    # Dist between (1,0,...) and (-1,0,...) is 2. dist^2 = 4.
    # Dist between (1,0,...) and (0,1,...) is 1. dist^2 = 2.
    # So 22 points of norm^2=1 have min dist^2 = 2. Valid.
    # But 22 < 220.
    
    # What if we mix norm^2=1 and norm^2=2?
    # Max norm^2 = 2. We need min dist^2 >= 2.
    # Take a norm^2=1 point p=(1,0,...) and a norm^2=2 point q=(1,1,0,...).
    # p.q = 1. dist^2 = 1 + 2 - 2(1) = 1.
    # 1 < 2. Invalid.
    # So we cannot mix norm^2=1 and norm^2=2 points if the norm^2=2 point shares an index with the norm^2=1 point.
    # To have dist^2 >= 2, we need p.q <= (1+2-2)/2 = 0.5. Since p.q is integer, p.q <= 0.
    # So p and q must be orthogonal.
    # p = (1,0,...). q must have 0 in the first position.
    # So we can take all 22 norm^2=1 points, and all norm^2=2 points that are orthogonal to each?
    # No, we need the set to be valid as a whole.
    # If we take all 220 norm^2=2 points, we cannot add any norm^2=1 point because each norm^2=1 point shares an index with some norm^2=2 points, leading to dist^2 = 1.
    
    # So the best we can do with integer coordinates and norm^2=2 is 220 points.
    
    # Let's verify the 220 points.
    points = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
    
    points = np.array(points, dtype=np.int64)
    
    # Check validity
    if points.shape[1] != 11:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    k = points.shape[0]
    if k < 2:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        # Remove zero vectors
        points = points[~np.all(points == 0, axis=1)]
    
    # Check min dist^2 >= max norm^2
    norms = np.sum(points**2, axis=1)
    max_norm = np.max(norms)
    
    min_dist = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist2 = np.sum(diff**2)
            if dist2 < min_dist:
                min_dist = dist2
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if min_dist >= max_norm:
        return points
    else:
        # Fallback to simple 2-point solution
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)


# EVOLVE-BLOCK-END
