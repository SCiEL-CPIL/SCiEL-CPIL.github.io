"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from math import sqrt

N = 64

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 320  # safety margin before 330s kill
    best_points = []
    best_count = 0

    # Precompute all grid points
    all_points = [(x, y) for x in range(N) for y in range(N)]

    # We want no three points forming an isosceles triangle.
    # This means: for any three distinct points A, B, C, the triangle is NOT isosceles.
    # An isosceles triangle has at least two equal sides. So we need to avoid:
    #   d(A,B) == d(A,C), d(A,B) == d(B,C), or d(A,C) == d(B,C)
    # for any three distinct points.
    
    # Key insight: A sufficient condition is that all pairwise squared distances are distinct
    # (a Sidon set in the distance metric). But that's very restrictive.
    
    # A better approach: we just need no three points to form an isosceles triangle.
    # This is equivalent to saying that for each point P, the distances from P to all other points
    # in the set are all distinct. AND for each pair of points (A, B), there is no other point C
    # such that d(A,C) = d(B,C) (which would make triangle ABC isosceles with AC=BC).
    
    # Actually, let's think more carefully. Triangle ABC is isosceles if any two of its three
    # side lengths are equal. The three sides are AB, AC, BC. So:
    # - AB = AC means A is the apex, and B, C are equidistant from A
    # - AB = BC means B is the apex, and A, C are equidistant from B  
    # - AC = BC means C is the apex, and A, B are equidistant from C
    
    # So the condition is: for every point P in the set, and every other pair of points (Q, R),
    # d(P,Q) != d(P,R). This means: from each point, all distances to other points are distinct.
    # This is necessary and sufficient for no isosceles triangles.
    
    # So we need a set where each point sees all other points at distinct distances.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
        
        random.shuffle(all_points)
        
        points = []
        
        # For each point in the set, we track the set of distances to other points
        # point_dists[i] = set of squared distances from points[i] to all other points in the set
        
        for p in all_points:
            if time.time() > deadline:
                break
            
            px, py = p
            
            # Check if adding p would create an isosceles triangle
            # For each existing point q, d(p,q) must not equal d(q, r) for any other existing point r
            # AND for p, all d(p, q) for existing q must be distinct
            # AND for each existing q, d(q,p) must not equal any existing d(q, r)
            
            valid = True
            dists_from_p = []
            
            for i, (qx, qy) in enumerate(points):
                dx = px - qx
                dy = py - qy
                d2 = dx * dx + dy * dy
                dists_from_p.append((d2, i))
            
            # Check 1: all distances from p to existing points are distinct
            seen = set()
            for d2, i in dists_from_p:
                if d2 in seen:
                    valid = False
                    break
                seen.add(d2)
            
            if not valid:
                continue
            
            # Check 2: for each existing point i, d(points[i], p) must not equal d(points[i], j) for any j != i
            # d(points[i], p) = d2_i (the distance from p to points[i])
            # We need d2_i not in {d(points[i], j) for all existing j}
            
            for i, (d2_i, _) in enumerate(dists_from_p):
                # Check if d2_i equals the distance from points[i] to any other existing point
                qx, qy = points[i]
                for j, (rx, ry) in enumerate(points):
                    if j == i:
                        continue
                    dx = qx - rx
                    dy = qy - ry
                    if dx * dx + dy * dy == d2_i:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # All checks passed, add p
            points.append(p)
            
            if len(points) > best_count:
                best_count = len(points)
                best_points = points[:]
                
                # If we've found a large set, we can try to extend it with local search
                if best_count >= 10:
                    # Try to find more points by scanning remaining grid
                    extended = points[:]
                    ext_valid = True
                    for rp in all_points:
                        if rp in extended:
                            continue
                        if time.time() > deadline - 10:
                            break
                        rx, ry = rp
                        valid_r = True
                        dists_r = []
                        for i, (sx, sy) in enumerate(extended):
                            dx = rx - sx
                            dy = ry - sy
                            d2 = dx * dx + dy * dy
                            dists_r.append((d2, i))
                        seen_r = set()
                        for d2, i in dists_r:
                            if d2 in seen_r:
                                valid_r = False
                                break
                            seen_r.add(d2)
                        if not valid_r:
                            continue
                        for i, (d2_i, _) in enumerate(dists_r):
                            sx, sy = extended[i]
                            for j, (tx, ty) in enumerate(extended):
                                if j == i:
                                    continue
                                dx = sx - tx
                                dy = sy - ty
                                if dx * dx + dy * dy == d2_i:
                                    valid_r = False
                                    break
                            if not valid_r:
                                break
                        if valid_r:
                            extended.append(rp)
                    
                    if len(extended) > best_count:
                        best_count = len(extended)
                        best_points = extended[:]
    
    if not best_points:
        # Fallback: return a small valid set
        best_points = [(0, 0), (1, 0), (2, 1), (4, 2), (7, 4)]
    
    return best_points
# EVOLVE-BLOCK-END
