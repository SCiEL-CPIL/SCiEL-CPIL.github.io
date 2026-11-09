import numpy as np
import time
from itertools import product

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580  # safety margin before 630s limit
    
    # Generate candidate points with coordinates in {-1, 0, 1}
    # We'll use a greedy approach to build a valid set
    # Constraint: min pairwise dist^2 >= max norm^2
    
    # Start with the known good set: points with exactly two non-zero entries (±1)
    # This gives 110 points, all with norm^2 = 2, min dist^2 = 2
    # Try to extend this or find a better configuration
    
    # Let's try a greedy approach with a broader candidate pool
    # Candidates: all vectors in {-1,0,1}^11 with norm^2 >= 1
    # But that's 3^11 = 177147 candidates, too many for pairwise checks
    
    # Instead, let's use the structure: if all points have norm^2 = c,
    # we need all pairwise distances >= c
    
    # Strategy: Use points with exactly 2 non-zero entries (the current approach)
    # and see if we can add more by being clever
    
    # Alternative: use a greedy algorithm with a smaller candidate set
    # Let's try: all vectors with exactly 2 non-zero entries from {-1,1}
    # That's what we have. Can we add vectors with 3 non-zero entries?
    
    # If I add a point with norm^2=3, then all existing points (norm^2=2) 
    # must be at distance >= 3 from it. And the new point must be >= 3 from other new points.
    # But existing points have norm^2=2, so max norm becomes 3.
    # Check: can a 3-nonzero point be at distance >= 3 from all 2-nonzero points?
    # E.g., (1,1,1,0,...) vs (1,1,0,...): dist^2 = 1 < 3. Fails.
    
    # So mixing different norm levels is hard. Let's stick with uniform norm.
    
    # What about using points with exactly 1 non-zero entry? norm^2=1, need dist>=1.
    # 22 points. Less than 110.
    
    # What about a different construction? 
    # Consider the D_11 lattice or similar.
    # Actually, let's try a greedy approach with a smart candidate ordering.
    
    # Generate all candidates with exactly 2 non-zero entries (±1)
    candidates = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Try to add more candidates: vectors with exactly 3 non-zero entries
    # But as analyzed, this is problematic. Let's verify with a greedy check.
    
    # Actually, let me try a different approach: use a greedy algorithm
    # where I sort candidates by some heuristic and add them if valid.
    
    # For now, let's just return the known 110-point solution
    # but verify it's correct
    points = np.array(candidates, dtype=np.int64)
    
    # Verify: check the constraint
    k = points.shape[0]
    norms = np.sum(points**2, axis=1)
    max_norm = np.max(norms)
    
    # Check pairwise distances
    valid = True
    for i in range(k):
        if time.time() > deadline:
            break
        for j in range(i+1, k):
            diff = points[i] - points[j]
            dist_sq = np.sum(diff**2)
            if dist_sq < max_norm:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: just two points
        pts = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        return pts
    
    return points
