"""Integral point sets in general position - search for 6+ points."""
import time
from math import isqrt, gcd
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_valid(pts):
    """Check all constraints for a set of integer points."""
    n = len(pts)
    if n < 3:
        return False
    # Check distinctness
    if len(set(pts)) != n:
        return False
    # Check all pairwise distances are integers
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    # Check no three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                    return False
    # Check no four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = pts[i], pts[j], pts[k], pts[l]
                    a11 = x1 - x2
                    a12 = y1 - y2
                    b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                    a21 = x1 - x3
                    a22 = y1 - y3
                    b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                    det = a11 * a22 - a21 * a12
                    if det == 0:
                        continue
                    A_num = b1 * a22 - b2 * a12
                    B_num = a11 * b2 - a21 * b1
                    C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                    lhs = x4 * A_num + y4 * B_num + C_num
                    rhs = -(x4*x4 + y4*y4) * det
                    if lhs == rhs:
                        return False
    return True

def _check_new_point_valid(existing_pts, new_pt):
    """Check if adding new_pt to existing_pts maintains all constraints."""
    n = len(existing_pts)
    x, y = new_pt
    
    # Check distinctness
    if new_pt in existing_pts:
        return False
    
    # Check distance to all existing points
    for i in range(n):
        dx = x - existing_pts[i][0]
        dy = y - existing_pts[i][1]
        d2 = dx * dx + dy * dy
        if not _is_square(d2):
            return False
    
    # Check no three collinear with new point
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = existing_pts[i]
            x2, y2 = existing_pts[j]
            if (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) == 0:
                return False
    
    # Check no four concyclic with new point
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                (x1, y1), (x2, y2), (x3, y3) = existing_pts[i], existing_pts[j], existing_pts[k]
                a11 = x1 - x2
                a12 = y1 - y2
                b1 = (x2*x2 + y2*y2) - (x1*x1 + y1*y1)
                a21 = x1 - x3
                a22 = y1 - y3
                b2 = (x3*x3 + y3*y3) - (x1*x1 + y1*y1)
                det = a11 * a22 - a21 * a12
                if det == 0:
                    continue
                A_num = b1 * a22 - b2 * a12
                B_num = a11 * b2 - a21 * b1
                C_num = -(x1*x1 + y1*y1) * det - x1 * A_num - y1 * B_num
                lhs = x * A_num + y * B_num + C_num
                rhs = -(x*x + y*y) * det
                if lhs == rhs:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Seed 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = seed_points if _check_valid(seed_points) else None
    
    # Generate Pythagorean pairs (a, b) with a²+b² = c², sorted by c
    pythagorean = []
    for a in range(1, 500):
        for b in range(1, 500):
            s = a * a + b * b
            c = isqrt(s)
            if c * c == s:
                pythagorean.append((a, b, c))
    
    # Sort by c to find smaller solutions first
    pythagorean.sort(key=lambda x: x[2])
    
    # Limit search for time
    max_c = 400
    pythagorean_small = [p for p in pythagorean if p[2] <= max_c]
    
    n_small = len(pythagorean_small)
    
    # Strategy: For 6 points with y-symmetry: (±a1, b1), (±a2, b2), (±a3, b3)
    # Constraints:
    # - a_i² + b_i² = c_i² (integer) for each i
    # - (a_i - a_j)² + (b_i - b_j)² = integer² for all i,j
    # - (a_i + a_j)² + (b_i - b_j)² = integer² for all i,j  [cross distances]
    # - No three collinear, no four concyclic
    
    found_6 = None
    checked = 0
    
    for i in range(n_small):
        if time.time() > deadline:
            break
        a1, b1, c1 = pythagorean_small[i]
        for j in range(i + 1, n_small):
            if time.time() > deadline:
                break
            a2, b2, c2 = pythagorean_small[j]
            # Check distance between (a1,b1) and (a2,b2)
            d2_same = (a1 - a2)**2 + (b1 - b2)**2
            if not _is_square(d2_same):
                continue
            # Check distance between (a1,b1) and (-a2,b2)
            d2_cross = (a1 + a2)**2 + (b1 - b2)**2
            if not _is_square(d2_cross):
                continue
            
            for k in range(j + 1, n_small):
                a3, b3, c3 = pythagorean_small[k]
                # Check (a1,b1) to (a3,b3)
                d2_13_same = (a1 - a3)**2 + (b1 - b3)**2
                if not _is_square(d2_13_same):
                    continue
                # Check (a1,b1) to (-a3,b3)
                d2_13_cross = (a1 + a3)**2 + (b1 - b3)**2
                if not _is_square(d2_13_cross):
                    continue
                # Check (a2,b2) to (a3,b3)
                d2_23_same = (a2 - a3)**2 + (b2 - b3)**2
                if not _is_square(d2_23_same):
                    continue
                # Check (a2,b2) to (-a3,b3)
                d2_23_cross = (a2 + a3)**2 + (b2 - b3)**2
                if not _is_square(d2_23_cross):
                    continue
                
                # All 6 distances check out! Now verify full constraints
                pts = [(a1, b1), (-a1, b1), (a2, b2), (-a2, b2), (a3, b3), (-a3, b3)]
                if _check_valid(pts):
                    found_6 = pts
                    break
            if found_6:
                break
        if found_6:
            break
    
    if found_6:
        best = found_6
    elif best is None:
        best = seed_points
    
    # Try to extend to 7 or 8 points if we found 6+
    if best and len(best) >= 6:
        # Try adding more points using incremental validation
        candidates = []
        for a in range(1, 200):
            for b in range(1, 200):
                s = a*a + b*b
                c = isqrt(s)
                if c*c != s:
                    continue
                for sign_x in [1, -1]:
                    for sign_y in [1, -1]:
                        pt = (sign_x * a, sign_y * b)
                        if pt not in best:
                            candidates.append(pt)
        
        # Sort candidates by some heuristic (e.g., distance to centroid)
        cx = sum(p[0] for p in best) / len(best)
        cy = sum(p[1] for p in best) / len(best)
        candidates.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for pt in candidates:
            if time.time() > deadline:
                break
            if _check_new_point_valid(best, pt):
                best = best + [pt]
    
    # Format output
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
