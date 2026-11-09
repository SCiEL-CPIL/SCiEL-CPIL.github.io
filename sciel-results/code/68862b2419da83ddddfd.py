"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _check_pairwise_distances_rational(points):
    """Check all pairwise squared distances are rational squares (for D=1, integer)."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    return True

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                if det == 0:
                    return False
    return True

def _check_no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    # Four points concyclic iff the determinant is zero
                    # Using the circle equation: x^2 + y^2 + Dx + Ey + F = 0
                    # For each point: xi^2+yi^2 + D*xi + E*yi + F = 0
                    # The 4x4 determinant must be nonzero for non-concyclic
                    a1, b1, c1 = x1*x1 + y1*y1, x1, y1
                    a2, b2, c2 = x2*x2 + y2*y2, x2, y2
                    a3, b3, c3 = x3*x3 + y3*y3, x3, y3
                    a4, b4, c4 = x4*x4 + y4*y4, x4, y4
                    # det of [[a1,b1,c1,1],[a2,b2,c2,1],[a3,b3,c3,1],[a4,b4,c4,1]]
                    # Expand along last column:
                    # det = a1*(b2*c3 - c2*b3 - b2*c4 + c2*b4 + b3*c4 - c3*b4)
                    #     - a2*(b1*c3 - c1*b3 - b1*c4 + c1*b4 + b3*c4 - c3*b4)
                    #     + a3*(b1*c2 - c1*b2 - b1*c4 + c1*b4 + b2*c4 - c2*b4)
                    #     - a4*(b1*c2 - c1*b2 - b1*c3 + c1*b3 + b2*c3 - c2*b3)
                    det = (
                        a1 * (b2*c3 - c2*b3 - b2*c4 + c2*b4 + b3*c4 - c3*b4)
                        - a2 * (b1*c3 - c1*b3 - b1*c4 + c1*b4 + b3*c4 - c3*b4)
                        + a3 * (b1*c2 - c1*b2 - b1*c4 + c1*b4 + b2*c4 - c2*b4)
                        - a4 * (b1*c2 - c1*b2 - b1*c3 + c1*b3 + b2*c3 - c2*b3)
                    )
                    if det == 0:
                        return False
    return True

def _validate(points):
    """Full validation of a point set."""
    if len(points) < 3:
        return False
    # Check distinctness
    seen = set()
    for p in points:
        if p in seen:
            return False
        seen.add(p)
    return (_check_pairwise_distances_rational(points) and
            _check_no_three_collinear(points) and
            _check_no_four_concyclic(points))

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Known 5-point set from seed
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find 6-point sets by extending known configurations
    # Eberhard's 6-point set (from literature):
    # A known 6-point integral point set in general position:
    eberhard_6 = [(0, 0), (3, 4), (4, 0), (5, 12), (15, 8), (9, 12)]
    if _validate(eberhard_6):
        pts = eberhard_6
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
    
    # Another known 6-point configuration
    # Try symmetric configurations
    # 6-point set: (0,0), (15,8), (9,12), (3,4), (4,0), (5,12) - same as above reordered
    
    # Search for 6-point sets by adding points to the 5-point seed
    base = seed_points
    best_pts = base
    
    # Try adding a 6th point
    # We need (x,y) such that distance to each of the 5 points is an integer
    # and no three collinear, no four concyclic
    
    # Precompute squared distances from each base point to origin for reference
    # Search in a bounded region
    search_radius = 200
    found_6 = None
    
    for x in range(-search_radius, search_radius + 1):
        if time.time() > deadline:
            break
        for y in range(-search_radius, search_radius + 1):
            if time.time() > deadline:
                break
            candidate = (x, y)
            # Check it's not already in base
            if candidate in base:
                continue
            # Check all distances to base points are integers
            ok = True
            for bx, by in base:
                dx = x - bx
                dy = y - by
                d2 = dx * dx + dy * dy
                if not _is_square(d2):
                    ok = False
                    break
            if not ok:
                continue
            # Check no three collinear
            new_set = base + [candidate]
            if not _check_no_three_collinear(new_set):
                continue
            # Check no four concyclic
            if not _check_no_four_concyclic(new_set):
                continue
            found_6 = new_set
            break
        if found_6:
            break
    
    if found_6:
        best_pts = found_6
        # Try to extend to 7 points
        if time.time() < deadline - 10:
            found_7 = None
            search_radius_7 = 300
            for x in range(-search_radius_7, search_radius_7 + 1):
                if time.time() > deadline:
                    break
                for y in range(-search_radius_7, search_radius_7 + 1):
                    if time.time() > deadline:
                        break
                    candidate = (x, y)
                    if candidate in found_6:
                        continue
                    ok = True
                    for bx, by in found_6:
                        dx = x - bx
                        dy = y - by
                        d2 = dx * dx + dy * dy
                        if not _is_square(d2):
                            ok = False
                            break
                    if not ok:
                        continue
                    new_set = found_6 + [candidate]
                    if not _check_no_three_collinear(new_set):
                        continue
                    if not _check_no_four_concyclic(new_set):
                        continue
                    found_7 = new_set
                    break
                if found_7:
                    break
            if found_7:
                best_pts = found_7
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_pts]}
# EVOLVE-BLOCK-END
