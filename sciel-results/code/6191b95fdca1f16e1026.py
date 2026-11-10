# EVOLVE-BLOCK-START
import numpy as np

def kissing_number11() -> np.ndarray:
    """
    Constructs points from the E8 ⊕ A3 lattice in Z^11 with squared norm 2.
    
    E8 roots (240 points) embedded in first 8 dimensions:
    - Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) — 28*4 = 112
    - Type 2: (±1/2, ±1/2, ..., ±1/2) — but these aren't integer!
    
    Wait, E8 roots in Z^8 are:
    - All permutations of (±1, ±1, 0, 0, 0, 0, 0, 0): C(8,2)*4 = 112
    - All vectors with all entries ±1/2: 2^8 = 256... but these aren't integers!
    
    Actually, the standard E8 lattice roots that are in Z^8:
    The E8 lattice is not a subset of Z^8 with just integer coordinates in the standard basis.
    The roots of E8 include half-integer vectors.
    
    Let me reconsider. The E8 root system consists of:
    1. Vectors with all coordinates ±1/2 (256 vectors)
    2. Vectors of form (±1, ±1, 0, ..., 0) (112 vectors)
    Total: 368? No, that's not right either.
    
    Actually, E8 has 240 roots:
    - 112 of form (±1, ±1, 0, 0, 0, 0, 0, 0)
    - 128 of form (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of minus signs
    
    But these half-integer vectors are NOT in Z^8!
    
    So we can't use them directly. We need integer points only.
    
    Let me think differently. In Z^11, what are all vectors with squared norm 2?
    They must have exactly two ±1 entries and rest 0. That gives C(11,2)*4 = 55*4 = 220.
    
    Are there other integer vectors with squared norm 2? 
    - One entry ±√2? No, must be integer.
    - Two entries: possibilities are (±1, ±1) giving norm²=2. That's it for integers.
    
    So in Z^11, the ONLY vectors with squared norm 2 are the weight-2 vectors.
    Maximum is 220.
    
    But wait — the condition is min_dist_sq >= max_norm_sq. If max_norm_sq = 2, we need min_dist_sq >= 2.
    
    What if we use points with squared norm 1? Then max_norm_sq = 1, and we need min_dist_sq >= 1.
    Points with norm² = 1: unit vectors ±e_i, giving 22 points.
    Min distance between distinct unit vectors: if same direction but opposite sign, dist² = 4. If different directions, dist² = 2. So min_dist_sq = 2 >= 1. ✓
    But only 22 points.
    
    What about mixing? Use some norm²=1 and some norm²=2 points?
    max_norm_sq would be 2, so we need min_dist_sq >= 2.
    Distance from a norm²=1 point to a norm²=2 point: could be 1 (e.g., e1 and e1+e2: dist = |e2| = 1, dist² = 1 < 2). ✗
    
    So we can't mix norms if max_norm_sq = 2.
    
    What if all points have norm² = 1? Only 22 points.
    
    What if we scale up? Use points with norm² = 4?
    Vectors with norm² = 4 in Z^11:
    - (±2, 0, ..., 0): 22 points
    - (±1, ±1, ±1, ±1, 0, ...): C(11,4)*16 = 330*16 = 5280 points
    
    Max_norm_sq = 4. Need min_dist_sq >= 4.
    
    For two (±2,0,...) points: if same index, dist² = 0 (same point) or 16 (opposite). If different indices, dist² = 4+4=8 >= 4. ✓
    
    For two weight-4 vectors: e.g., (1,1,1,1,0,...) and (1,1,1,-1,0,...): dist² = (0)²+(0)²+(0)²+(2)² = 4. ✓
    (1,1,1,1,0,...) and (1,1,1,0,1,0,...): dist² = 0+0+0+(1)²+(1)² = 2 < 4. ✗
    
    So we can't just take all weight-4 vectors.
    
    This is getting complex. Let me go back to the 220-point solution which is valid.
    
    Actually, re-reading the task: "the minimum squared distance over all pairs of points is >= the maximum squared norm of the points"
    
    With 220 weight-2 vectors, max_norm_sq = 2, min_dist_sq = 2. Valid.
    
    I'll stick with the 220-point solution as it's provably valid and optimal for norm²=2 integer vectors.
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
