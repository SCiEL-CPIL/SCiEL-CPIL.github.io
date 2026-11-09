import numpy as np
import time

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 600  # wall-clock limit
    safety_margin = 30  # seconds to leave before timeout

    points = []

    # Phase 1: All vectors of form ±e_i ± e_j for i < j (4 per pair)
    # This gives 4 * C(11,2) = 4 * 55 = 220 points
    for i in range(d):
        for j in range(i + 1, d):
            v1 = np.zeros(d, dtype=np.int64)
            v1[i] = 1; v1[j] = 1
            points.append(v1)
            v2 = np.zeros(d, dtype=np.int64)
            v2[i] = 1; v2[j] = -1
            points.append(v2)
            v3 = np.zeros(d, dtype=np.int64)
            v3[i] = -1; v3[j] = 1
            points.append(v3)
            v4 = np.zeros(d, dtype=np.int64)
            v4[i] = -1; v4[j] = -1
            points.append(v4)

    # Phase 2: Try to add single-unit vectors ±e_i (11 + 11 = 22 points)
    # Check if they satisfy the constraint with existing points
    # For a point p with norm^2 = 2, distance to e_i:
    # If p has component at i: |p_i - 1|^2 + sum_{j!=i} p_j^2 = 1 + (norm^2 - p_i^2) = 1 + 2 - 1 = 2
    # If p has component 0 at i: 1^2 + 2 = 3
    # If p has component -1 at i: (-1-1)^2 + (2-1) = 4 + 1 = 5
    # So min distance from e_i to the 220 points is 2, and norm^2 of e_i is 1.
    # But we need min pair distance >= max norm^2. Max norm^2 would become 2.
    # Min pair distance between e_i and points that have 0 at i: 3 >= 2 OK
    # Min pair distance between e_i and points that have 1 at i: 2 >= 2 OK
    # Min pair distance between e_i and e_j: 2 >= 2 OK
    # So we can add all 22 unit vectors!
    
    for i in range(d):
        vi = np.zeros(d, dtype=np.int64)
        vi[i] = 1
        points.append(vi)
    
    for i in range(d):
        vi = np.zeros(d, dtype=np.int64)
        vi[i] = -1
        points.append(vi)

    # Phase 3: Try to add vectors of form 2*e_i (norm^2 = 4)
    # Distance from 2*e_i to e_i: (2-1)^2 = 1 < 4 -> NO
    # Distance from 2*e_i to e_j (j!=i): 4 + 1 = 5 >= 4 OK
    # Distance from 2*e_i to points with component at i:
    #   If p_i = 1: (2-1)^2 + rest = 1 + (2-1) = 2 < 4 -> NO
    # So 2*e_i cannot be added.

    # Phase 4: Try vectors with three non-zero components
    # e.g., (1,1,1,0,...,0) norm^2 = 3
    # Distance to (1,1,0,0,...): (0)^2+(0)^2+(1)^2 = 1 < 3 -> NO
    # So these don't work.

    # Phase 5: Try to add more from the 220 set - we already have all 220.
    # What about vectors like (1,1,1,-1,0,...) norm^2 = 4?
    # Distance to (1,1,0,0,...): 0+0+1+1 = 2 < 4 -> NO

    # Let's verify: can we add vectors of norm^2 = 1 that aren't unit vectors?
    # In integers, norm^2 = 1 means exactly one component is ±1, rest 0. We have all of those.

    # What about the negative of the 220? We already include both signs in the 4 per pair.
    # (1,1), (1,-1), (-1,1), (-1,-1) - yes, all four sign combos are included.

    # Let's try a different approach: can we scale?
    # If all points have norm^2 = 2, and we scale by 2, norm^2 = 8, min dist^2 = 8. That works!
    # But we'd just have 220 * 2 = 440 points? No, scaling gives different points.
    # Actually, if we take the 220 points AND 2*those 220 points, that's 440 points.
    # But distance between a point p (norm^2=2) and 2p: |p - 2p|^2 = |−p|^2 = 2 < 8 -> NO

    # What about adding the negatives? We already have them.

    # Let me think about this differently. The constraint is:
    # min_{p,q in S, p!=q} ||p-q||^2 >= max_{p in S} ||p||^2
    # 
    # For our 220+22=242 points, max norm^2 = 2, min pair dist^2 = 2.
    #
    # Can we add more points with norm^2 <= 2? In integers, that means norm^2 in {1, 2}.
    # norm^2 = 1: unit vectors (22 total, already included)
    # norm^2 = 2: vectors with two ±1 components (220 total, already included)
    #
    # So we have ALL integer vectors with norm^2 <= 2 in R^11! That's 22 + 220 = 242.
    #
    # Can we add any point with norm^2 = 3? Need distance >= 3 to all existing points.
    # A point with norm^2 = 3 has three non-zero components (each ±1) or one component ±√3 (not integer).
    # So it must have three ±1 entries. E.g., (1,1,1,0,...,0).
    # Distance to (1,1,0,0,...): 0^2+0^2+1^2 = 1 < 3. FAIL.
    #
    # What about norm^2 = 4 with one component ±2?
    # (2,0,...,0): distance to (1,0,...,0) = 1 < 4. FAIL.
    #
    # So it seems 242 is the maximum for integer points.

    points_array = np.array(points, dtype=np.int64)
    points_array = np.around(points_array).astype(np.int64)
    
    return points_array
