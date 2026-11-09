"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _dist2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _all_int_distances(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_square(_dist2(points[i], points[j])):
                return False
    return True

def _are_collinear(p1, p2, p3):
    # Cross product of (p2-p1) and (p3-p1)
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    return True

def _are_concyclic(p1, p2, p3, p4):
    # Four points are concyclic iff the determinant of the circle equation is zero.
    # Circle: x^2 + y^2 + Dx + Ey + F = 0
    # For point (x,y): x^2+y^2 + Dx + Ey + F = 0
    # Determinant of the 4x4 system must be 0
    a1 = p1[0]; b1 = p1[1]; c1 = a1*a1 + b1*b1
    a2 = p2[0]; b2 = p2[1]; c2 = a2*a2 + b2*b2
    a3 = p3[0]; b3 = p3[1]; c3 = a3*a3 + b3*b3
    a4 = p4[0]; b4 = p4[1]; c4 = a4*a4 + b4*b4
    # det of [[a,b,c,1],[a,b,c,1],[a,b,c,1],[a,b,c,1]]
    # Use expansion: 
    # | a1 b1 c1 1 |
    # | a2 b2 c2 1 |
    # | a3 b3 c3 1 |
    # | a4 b4 c4 1 |
    # Subtract row1 from rows 2,3,4 to simplify:
    # | a1 b1 c1 1 |
    # | a2-a1 b2-b1 c2-c1 0 |
    # | a3-a1 b3-b1 c3-c1 0 |
    # | a4-a1 b4-b1 c4-c1 0 |
    # = 1 * det of 3x3:
    # | a2-a1 b2-b1 c2-c1 |
    # | a3-a1 b3-b1 c3-c1 |
    # | a4-a1 b4-b1 c4-c1 |
    d1 = a2 - a1; e1 = b2 - b1; f1 = c2 - c1
    d2 = a3 - a1; e2 = b3 - b1; f2 = c3 - c1
    d3 = a4 - a1; e3 = b4 - b1; f3 = c4 - c1
    det = d1*(e2*f3 - e3*f2) - e1*(d2*f3 - d3*f2) + f1*(d2*e3 - d3*e2)
    return det == 0

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _is_valid(points):
    if not _all_int_distances(points):
        return False
    if not _no_three_collinear(points):
        return False
    if not _no_four_concyclic(points):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = base_points[:]
    
    # Verify base is valid
    assert _is_valid(best), "Base configuration should be valid"
    
    # Try to add a 6th point by grid search
    # The existing points span x in [-56, 16], y in [0, 63]
    # Search in a reasonable range around this
    x_min = -120
    x_max = 120
    y_min = -60
    y_max = 120
    
    # First, precompute distances from base points to speed up checking
    # For a candidate point p, we need:
    # 1. dist2(p, pi) is a perfect square for all i
    # 2. p is not collinear with any pair of base points
    # 3. p is not concyclic with any triple of base points
    
    found_six = False
    for x in range(x_min, x_max + 1):
        if time.time() > deadline:
            break
        for y in range(y_min, y_max + 1):
            if time.time() > deadline:
                break
            p = (x, y)
            # Skip if already in base
            if p in base_points:
                continue
            # Check integer distances to all base points
            ok = True
            for bp in base_points:
                if not _is_square(_dist2(p, bp)):
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear: p with any pair from base
            for i in range(len(base_points)):
                for j in range(i + 1, len(base_points)):
                    if _are_collinear(p, base_points[i], base_points[j]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check no four concyclic: p with any triple from base
            for i in range(len(base_points)):
                for j in range(i + 1, len(base_points)):
                    for k in range(j + 1, len(base_points)):
                        if _are_concyclic(p, base_points[i], base_points[j], base_points[k]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Found a valid 6th point!
            new_points = base_points + [p]
            if _is_valid(new_points):
                best = new_points
                found_six = True
                break
        if found_six:
            break
    
    # If we found 6 points, try to find a 7th
    if len(best) == 6 and time.time() < deadline - 5:
        # Try adding a 7th point
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
            for y in range(y_min, y_max + 1):
                if time.time() > deadline:
                    break
                p = (x, y)
                if p in best:
                    continue
                ok = True
                for bp in best:
                    if not _is_square(_dist2(p, bp)):
                        ok = False
                        break
                if not ok:
                    continue
                for i in range(len(best)):
                    for j in range(i + 1, len(best)):
                        if _are_collinear(p, best[i], best[j]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                for i in range(len(best)):
                    for j in range(i + 1, len(best)):
                        for k in range(j + 1, len(best)):
                            if _are_concyclic(p, best[i], best[j], best[k]):
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                new_points = best + [p]
                if _is_valid(new_points):
                    best = new_points
                    break
            if len(best) == 7:
                break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}

# EVOLVE-BLOCK-END
