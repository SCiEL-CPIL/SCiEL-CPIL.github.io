import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = 600.0
    
    # Initial set: 11 e_i and 55 (-e_a - e_b)
    points = []
    for i in range(d):
        v = np.zeros(d, dtype=np.int64)
        v[i] = 1
        points.append(v)
        
    for a in range(d):
        for b in range(a + 1, d):
            v = np.zeros(d, dtype=np.int64)
            v[a] = -1
            v[b] = -1
            points.append(v)
            
    points = np.array(points)
    
    # Helper to check if a point can be added
    def can_add(p, current_points):
        # p is 1-D array
        # current_points is 2-D array
        if np.any(p == 0):
            return False
        # Check norm
        norm_sq = np.dot(p, p)
        # Check distance to all existing points
        # dist_sq = ||p - q||^2 = ||p||^2 + ||q||^2 - 2 p.q
        # We need min dist_sq >= max_norm_sq
        # max_norm_sq is current max norm in set (which is 2 for our initial set)
        # But if we add points with larger norm, max_norm_sq increases.
        # So we must check against the NEW max norm.
        # Let N_max_new = max(current_max_norm, norm_sq)
        # We need ||p - q||^2 >= N_max_new for all q.
        
        current_max_norm = np.max(np.sum(current_points**2, axis=1))
        N_max_new = max(current_max_norm, norm_sq)
        
        for q in current_points:
            diff = p - q
            dist_sq = np.dot(diff, diff)
            if dist_sq < N_max_new:
                return False
        return True

    # Candidate pool
    # Generate all vectors with coordinates in {-2, -1, 0, 1, 2} and norm squared <= 4
    # This covers weight 1, 2, 3, 4 (with small weights)
    # Actually, let's just generate weight 1, 2, 3 vectors with signs.
    candidates = []
    seen = set()
    
    # Add existing points to seen
    for p in points:
        seen.add(tuple(p))
        
    # Generate candidates
    # Weight 1
    for i in range(d):
        for s in [1, -1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = s
            if tuple(v) not in seen:
                candidates.append(v)
                seen.add(tuple(v))
                
    # Weight 2
    for a in range(d):
        for b in range(a + 1, d):
            for sa in [1, -1]:
                for sb in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[a] = sa
                    v[b] = sb
                    if tuple(v) not in seen:
                        candidates.append(v)
                        seen.add(tuple(v))
                        
    # Weight 3
    for a in range(d):
        for b in range(a + 1, d):
            for c in range(b + 1, d):
                for sa in [1, -1]:
                    for sb in [1, -1]:
                        for sc in [1, -1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[a] = sa
                            v[b] = sb
                            v[c] = sc
                            if tuple(v) not in seen:
                                candidates.append(v)
                                seen.add(tuple(v))
                                
    # Weight 4 (maybe)
    # Skip for now to keep candidate list manageable.
    
    # Shuffle candidates to avoid bias?
    np.random.shuffle(candidates)
    
    # Greedy addition
    changed = True
    while changed:
        changed = False
        for i, c in enumerate(candidates):
            if can_add(c, points):
                points = np.vstack([points, c])
                candidates.pop(i)
                changed = True
                if time.time() > deadline - 10:
                    break
        if time.time() > deadline - 10:
            break
            
    return points
