# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def _e8_roots():
    """Generate all 240 roots of the E8 lattice in Z^8."""
    roots = set()
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) - C(8,2)*4 = 112
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                r = [0]*8
                r[i] = si
                r[j] = sj
                roots.add(tuple(r))
    # Type 2: (±1/2, ±1/2, ..., ±1/2) with even number of minus signs
    # But these are NOT in Z^8! They have half-integer coordinates.
    # E8 roots in the standard embedding: 240 roots total
    # The 112 above are the "coordinate" roots
    # The other 128 are half-integer: (±1/2)^8 with even number of minus signs
    # But we need INTEGER coordinates!
    
    # Actually, the E8 lattice can be represented with integer coordinates.
    # The standard E8 root system has 240 roots, all of which can be written with integer coordinates
    # in a suitable basis. But in the standard coordinate representation,
    # 112 roots are (±1,±1,0,...,0) and 128 are (±1/2,...,±1/2) with even minus signs.
    
    # Since we need integer coordinates, we can only use the 112 integer roots.
    # Let's verify: are there other integer representations?
    
    # Actually, the problem states points must have integer coordinates (cast to int64).
    # So we can only use roots that have all-integer coordinates in Z^8.
    # That gives us only 112 roots from E8, not 240.
    
    return list(roots)

def _a3_roots_3d():
    """Generate all integer vectors in Z^3 with squared norm 2."""
    roots = set()
    for i, j in combinations(range(3), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                r = [0]*3
                r[i] = si
                r[j] = sj
                roots.add(tuple(r))
    return list(roots)

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    Strategy: Combine E8 integer roots (112) in dims 0-7 with A3-type roots (12) in dims 8-10.
    All points have norm^2 = 2.
    Cross-distances between E8-part and A3-part points: dist^2 = 2 + 2 = 4 >= 2.
    Within E8-part: min dist^2 = 2 (verified for the 112 integer roots).
    Within A3-part: min dist^2 = 2 (same structure as weight-2 vectors).
    
    Total: 112 + 12 = 124 points. This is LESS than 220, so this approach alone is worse.
    
    Alternative: Use ALL weight-2 vectors in Z^11 (220 points) as before, which is better.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    d = 11
    points_list = []
    
    # Generate all vectors with exactly 2 non-zero entries, each being +/- 1
    for i, j in combinations(range(d), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = si
                p[j] = sj
                points_list.append(p)
                
    if time.time() > deadline - 10:
         return np.array(points_list[:2], dtype=np.int64)

    points = np.array(points_list, dtype=np.int64)
    
    return points

# EVOLVE-BLOCK-END
