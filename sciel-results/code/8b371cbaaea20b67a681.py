import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_collinear(p1, p2, p3):
        # (x2-x1)*(y3-y1) == (x3-x1)*(y2-y1)
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1])

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        a1 = x1*x1 + y1*y1
        a2 = x2*x2 + y2*y2
        a3 = x3*x3 + y3*y3
        a4 = x4*x4 + y4*y4
        
        det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
            - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
            + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
            
        return det == 0

    def validate_set(points):
        n = len(points)
        if n < 3:
            return False
        # Check all pairwise distances are integer
        for i in range(n):
            for j in range(i+1, n):
                d = dist_sq(points[i], points[j])
                if not is_square(d):
                    return False
        # Check no three collinear
        for i, j, k in combinations(range(n), 3):
            if is_collinear(points[i], points[j], points[k]):
                return False
        # Check no four concyclic
        for i, j, k, l in combinations(range(n), 4):
            if is_concyclic(points[i], points[j], points[k], points[l]):
                return False
        return True

    # Start with a known 5-point set
    seed = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    if not validate_set(seed):
        # Fallback if seed is somehow invalid (should not happen)
        return {'D': 1, 'points': [[[0,1],[0,1]], [[3,1],[4,1]], [[0,1],[12,1]]]}

    best_set = list(seed)
    
    # Search for a 6th point in an expanding grid
    # We check integer coordinates first
    # Range: [-M, M]
    
    def find_6th_point(existing, M):
        n = len(existing)
        # Precompute squared distances between existing points? Not strictly needed for adding 1 point.
        # But we need to check dist to all existing.
        
        # Generate candidates
        for x in range(-M, M + 1):
            for y in range(-M, M + 1):
                p = (x, y)
                # Skip if already in set
                if p in existing:
                    continue
                
                # Check distances to all existing points
                valid_dist = True
                for q in existing:
                    d = dist_sq(p, q)
                    if not is_square(d):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                
                # Check collinearity with any pair of existing points
                valid_coll = True
                for i in range(n):
                    for j in range(i+1, n):
                        if is_collinear(p, existing[i], existing[j]):
                            valid_coll = False
                            break
                    if not valid_coll:
                        break
                if not valid_coll:
                    continue
                
                # Check concyclicity with any triple of existing points
                valid_conc = True
                if n >= 3:
                    for i, j, k in combinations(range(n), 3):
                        if is_concyclic(p, existing[i], existing[j], existing[k]):
                            valid_conc = False
                            break
                if not valid_conc:
                    continue
                
                return p
        return None

    # Try increasing box sizes
    # M=100 is fast. M=500 is still fast.
    for M in [100, 200, 500, 1000]:
        if time.time() > deadline - 5.0:
            break
        new_p = find_6th_point(best_set, M)
        if new_p:
            best_set.append(new_p)
            # Try to find a 7th point?
            # Unlikely, but let's try a small search
            new_p2 = find_6th_point(best_set, 100)
            if new_p2:
                best_set.append(new_p2)
            break

    # Final verification
    if not validate_set(best_set):
        # If the extended set is invalid, fall back to the seed
        if validate_set(seed):
            best_set = seed
        else:
            # Should not happen
            best_set = seed

    # Format output
    D = 1
    points_formatted = []
    for (x, y) in best_set:
        points_formatted.append([[x, 1], [y, 1]])

    return {'D': D, 'points': points_formatted}
