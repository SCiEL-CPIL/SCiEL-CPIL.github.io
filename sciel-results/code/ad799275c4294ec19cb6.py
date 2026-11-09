"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _rational_dist2(p1, p2, D):
    """Return (num, den) of squared distance between two points (x, y*sqrt(D))."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    # dist^2 = dx^2 + D * dy^2
    num = dx * dx + D * dy * dy
    den = 1
    return num, den

def _check_rational_dist(p1, p2, D):
    """Check if squared distance is a perfect square rational (i.e., distance is rational)."""
    num, den = _rational_dist2(p1, p2, D)
    if den == 1:
        return _is_square(num)
    # For rational distance, num/den must be a perfect square of a rational
    # i.e., num * den must be a perfect square (after reducing)
    g = math.gcd(num, den)
    num_r = num // g
    den_r = den // g
    return _is_square(num_r * den_r)

def _check_no_three_collinear(points, D):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Cross product: (p2-p1) x (p3-p1)
                dx1 = p2[0] - p1[0]
                dy1 = (p2[1] - p1[1]) * math.sqrt(D)  # We work with y*sqrt(D)
                dx2 = p3[0] - p1[0]
                dy2 = (p3[1] - p1[1]) * math.sqrt(D)
                # Cross product = dx1*dy2 - dy1*dx2
                # = dx1*(p3[1]-p1[1])*sqrt(D) - (p2[1]-p1[1])*sqrt(D)*dx2
                # = sqrt(D) * (dx1*(p3[1]-p1[1]) - (p2[1]-p1[1])*dx2)
                # So we just need to check if dx1*(p3[1]-p1[1]) - (p2[1]-p1[1])*dx2 == 0
                val = dx1 * (p3[1] - p1[1]) - (p2[1] - p1[1]) * dx2
                if val == 0:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # Four points are concyclic iff the determinant of the circle equation is 0
                    # Circle: x^2 + y^2 + Ax + By + C = 0 (in coordinates where y is actual y*sqrt(D))
                    # We use: x^2 + (y*sqrt(D))^2 + Ax + B*(y*sqrt(D)) + C = 0
                    # = x^2 + D*y^2 + Ax + B*sqrt(D)*y + C = 0
                    # For rational x,y and integer D, this works in Q(sqrt(D))
                    # The condition is that all four points satisfy the same circle equation.
                    # Use the determinant test:
                    # |x1^2 + D*y1^2, x1, y1, 1|
                    # |x2^2 + D*y2^2, x2, y2, 1| = 0
                    # |x3^2 + D*y3^2, x3, y3, 1|
                    # |x4^2 + D*y4^2, x4, y4, 1|
                    # But we need to be careful: the circle equation in (x, Y) where Y = y*sqrt(D) is
                    # x^2 + Y^2 + Ax + BY + C = 0
                    # So the determinant should use Y = y*sqrt(D).
                    # However, since we're working with rational y, Y = y*sqrt(D) is in Q(sqrt(D)).
                    # The determinant will be in Q(sqrt(D)). It's zero iff both rational and sqrt(D) parts are 0.
                    
                    # Let's compute using the fact that x^2 + D*y^2 is rational.
                    # The matrix has entries:
                    # Row i: [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
                    # The determinant is in Q(sqrt(D)).
                    # We can write det = A + B*sqrt(D) where A, B are rational.
                    # det = 0 iff A = 0 and B = 0.
                    
                    # Let's compute the determinant symbolically.
                    # Actually, let's use a different approach.
                    # Four points (x_i, Y_i) are concyclic iff:
                    # The determinant of [x_i^2 + Y_i^2, x_i, Y_i, 1] for i=1..4 is 0.
                    # Y_i = y_i * sqrt(D)
                    # x_i^2 + Y_i^2 = x_i^2 + D*y_i^2 (rational)
                    # So the matrix is:
                    # [x1^2 + D*y1^2, x1, y1*sqrt(D), 1]
                    # [x2^2 + D*y2^2, x2, y2*sqrt(D), 1]
                    # [x3^2 + D*y3^2, x3, y3*sqrt(D), 1]
                    # [x4^2 + D*y4^2, x4, y4*sqrt(D), 1]
                    
                    # Let a_i = x_i^2 + D*y_i^2, b_i = x_i, c_i = y_i
                    # Matrix: [a_i, b_i, c_i*sqrt(D), 1]
                    # det = det of 4x4 matrix.
                    # Expand along last column:
                    # det = 1 * det3([a1,b1,c1*sqrt(D); a2,b2,c2*sqrt(D); a3,b3,c3*sqrt(D)])
                    #     - 1 * det3([a1,b1,c1*sqrt(D); a2,b2,c2*sqrt(D); a4,b4,c4*sqrt(D)])
                    #     + 1 * det3([a1,b1,c1*sqrt(D); a3,b3,c3*sqrt(D); a4,b4,c4*sqrt(D)])
                    #     - 1 * det3([a2,b2,c2*sqrt(D); a3,b3,c3*sqrt(D); a4,b4,c4*sqrt(D)])
                    
                    # Each det3 is of the form:
                    # |a_i, b_i, c_i*sqrt(D)|
                    # |a_j, b_j, c_j*sqrt(D)|
                    # |a_k, b_k, c_k*sqrt(D)|
                    # = (a_i(b_j c_k - b_j c_k) - b_i(a_j c_k - a_j c_k) + c_i*sqrt(D)(a_j b_k - a_k b_j))
                    # Wait, let me be more careful.
                    # det = a_i*(b_j*c_k*sqrt(D) - b_j*c_k*sqrt(D)) - b_i*(a_j*c_k*sqrt(D) - a_j*c_k*sqrt(D)) + c_i*sqrt(D)*(a_j*b_k - a_k*b_j)
                    # Hmm, that's not right. Let me redo.
                    # |a_i, b_i, c_i*sqrt(D)|
                    # |a_j, b_j, c_j*sqrt(D)|
                    # |a_k, b_k, c_k*sqrt(D)|
                    # = a_i*(b_j*c_k*sqrt(D) - b_k*c_j*sqrt(D)) - b_i*(a_j*c_k*sqrt(D) - a_k*c_j*sqrt(D)) + c_i*sqrt(D)*(a_j*b_k - a_k*b_j)
                    # = sqrt(D) * [a_i*(b_j*c_k - b_k*c_j) - b_i*(a_j*c_k - a_k*c_j) + c_i*(a_j*b_k - a_k*b_j)]
                    
                    # So each 3x3 det is sqrt(D) times a rational number.
                    # Therefore the 4x4 det is:
                    # det = 1 * sqrt(D)*R1 - 1 * sqrt(D)*R2 + 1 * sqrt(D)*R3 - 1 * sqrt(D)*R4
                    # = sqrt(D) * (R1 - R2 + R3 - R4)
                    
                    # So det = 0 iff (R1 - R2 + R3 - R4) = 0 (since sqrt(D) != 0).
                    
                    # Let's compute R1, R2, R3, R4 where Rijk = a_i*(b_j*c_k - b_k*c_j) - b_i*(a_j*c_k - a_k*c_j) + c_i*(a_j*b_k - a_k*b_j)
                    
                    a1, b1, c1 = p1[0]**2 + D*p1[1]**2, p1[0], p1[1]
                    a2, b2, c2 = p2[0]**2 + D*p2[1]**2, p2[0], p2[1]
                    a3, b3, c3 = p3[0]**2 + D*p3[1]**2, p3[0], p3[1]
                    a4, b4, c4 = p4[0]**2 + D*p4[1]**2, p4[0], p4[1]
                    
                    def R(i, j, k):
                        ai, bi, ci = (a1, b1, c1)[i-1], (b1, b2, b3, b4)[i-1], (c1, c2, c3, c4)[i-1]
                        aj, bj, cj = (a1, b1, c1)[j-1], (b1, b2, b3, b4)[j-1], (c1, c2, c3, c4)[j-1]
                        ak, bk, ck = (a1, b1, c1)[k-1], (b1, b2, b3, b4)[k-1], (c1, c2, c3, c4)[k-1]
                        return ai*(bj*ck - bk*cj) - bi*(aj*ck - ak*cj) + ci*(aj*bk - ak*bj)
                    
                    det_val = R(1,2,3) - R(1,2,4) + R(1,3,4) - R(2,3,4)
                    if det_val == 0:
                        return False
    return True

def _is_valid(points, D):
    """Check all conditions: rational distances, no three collinear, no four concyclic."""
    n = len(points)
    if n < 3:
        return False
    # Check all pairwise distances are rational
    for i in range(n):
        for j in range(i + 1, n):
            if not _check_rational_dist(points[i], points[j], D):
                return False
    # Check no three collinear
    if not _check_no_three_collinear(points, D):
        return False
    # Check no four concyclic
    if n >= 4:
        if not _check_no_four_concyclic(points, D):
            return False
    return True

def _try_add_point(current_points, D, candidate):
    """Try to add a candidate point to the current set."""
    new_points = current_points + [candidate]
    return _is_valid(new_points, D)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    best_points = base_points
    
    # Try to find 6+ point sets by searching for additional points
    # Strategy: start with subsets of base points and try to add new points
    # For D=1, points are integer lattice points with integer pairwise distances
    
    # Precompute: for D=1, we need integer points with integer pairwise distances
    # Search in a bounded region
    
    max_coord = 200
    
    # Generate candidate points: integer points in range with at least one other point at integer distance
    # For efficiency, only consider points that have integer distance to at least one base point
    
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            # Check if this point has integer distance to at least one base point
            has_int_dist = False
            for bx, by in base_points:
                d2 = (x - bx)**2 + (y - by)**2
                if _is_square(d2):
                    has_int_dist = True
                    break
            if has_int_dist:
                candidates.append((x, y))
    
    # Try to build larger sets by greedy addition
    # Start with various subsets and try to extend
    import random
    random.seed(42)
    
    best_found = 5
    
    # Try greedy: start with base, try adding candidates one by one
    current = list(base_points)
    for cand in candidates:
        if (cand[0], cand[1]) in current:
            continue
        if time.time() > deadline:
            break
        if _try_add_point(current, D, cand):
            current.append(cand)
            if len(current) > best_found:
                best_found = len(current)
                best_points = list(current)
            if len(current) >= 8:
                break
    
    if best_found >= 6:
        points = best_points
    else:
        points = base_points
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END
