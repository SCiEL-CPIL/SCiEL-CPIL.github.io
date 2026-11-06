"""
Search for integral point sets in general position (no three collinear, no four concyclic)
with rational coordinates of the form (x, y*sqrt(D)) for squarefree D.
The score is the number of points (maximize).
"""

import time
import math
import random
from fractions import Fraction
from itertools import combinations

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 145.0  # seconds total (evaluator kills at 150s)
_SAFETY_MARGIN = 5.0  # return 5s before limit

def _time_left():
    return _TIME_LIMIT - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _is_square(n):
    """Check if a non-negative integer is a perfect square."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _sq_dist_is_square(p1, p2, D):
    """
    Check if squared distance between two points is a perfect square in Q.
    Points are (x1, y1*sqrt(D)), (x2, y2*sqrt(D)) where x,y are Fractions.
    Squared distance = (x1-x2)^2 + D*(y1-y2)^2
    We need this to be a perfect square of a rational number.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    sq_d = dx * dx + D * dy * dy
    if sq_d < 0:
        return False
    num = sq_d.numerator
    den = sq_d.denominator
    return _is_square(num) and _is_square(den)

def _dist_is_rational(p1, p2, D):
    """Check if distance between two points is rational."""
    return _sq_dist_is_square(p1, p2, D)

def _collinear_check(x1, y1, x2, y2, x3, y3):
    """Check if three points are collinear. Returns True if collinear."""
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _no_three_collinear(points):
    """Check no three points are collinear using exact arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear_check(points[i][0], points[i][1],
                                    points[j][0], points[j][1],
                                    points[k][0], points[k][1]):
                    return False
    return True

def _det4(m):
    """Compute 4x4 determinant with Fraction entries."""
    return (m[0][0] * (m[1][1] * m[2][2] * m[3][3] + m[1][2] * m[2][3] * m[3][1] + m[1][3] * m[2][1] * m[3][2]
                        - m[1][1] * m[2][3] * m[3][2] - m[1][2] * m[2][1] * m[3][3] - m[1][3] * m[2][2] * m[3][1])
            - m[0][1] * (m[1][0] * m[2][2] * m[3][3] + m[1][2] * m[2][3] * m[3][0] + m[1][3] * m[2][0] * m[3][2]
                         - m[1][0] * m[2][3] * m[3][2] - m[1][2] * m[2][0] * m[3][3] - m[1][3] * m[2][2] * m[3][0])
            + m[0][2] * (m[1][0] * m[2][1] * m[3][3] + m[1][1] * m[2][3] * m[3][0] + m[1][3] * m[2][0] * m[3][1]
                         - m[1][0] * m[2][3] * m[3][1] - m[1][1] * m[2][0] * m[3][3] - m[1][3] * m[2][1] * m[3][0])
            - m[0][3] * (m[1][0] * m[2][1] * m[3][2] + m[1][1] * m[2][2] * m[3][0] + m[1][2] * m[2][0] * m[3][1]
                         - m[1][0] * m[2][2] * m[3][1] - m[1][1] * m[2][0] * m[3][2] - m[1][2] * m[2][1] * m[3][0]))

def _concyclic_check(p1, p2, p3, p4, D):
    """Check if four points are concyclic. Returns True if concyclic."""
    a = [[p1[0] ** 2 + D * p1[1] ** 2, p1[0], p1[1], Fraction(1)],
         [p2[0] ** 2 + D * p2[1] ** 2, p2[0], p2[1], Fraction(1)],
         [p3[0] ** 2 + D * p3[1] ** 2, p3[0], p3[1], Fraction(1)],
         [p4[0] ** 2 + D * p4[1] ** 2, p4[0], p4[1], Fraction(1)]]
    return _det4(a) == 0

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for combo in combinations(range(n), 4):
        if _concyclic_check(points[combo[0]], points[combo[1]],
                            points[combo[2]], points[combo[3]], D):
            return False
    return True

def _validate(points, D):
    """Full validation of a point set."""
    n = len(points)
    if n < 3:
        return False
    # Check all pairwise distances are rational
    for i in range(n):
        for j in range(i + 1, n):
            if not _dist_is_rational(points[i], points[j], D):
                return False
    # Check no three collinear
    if not _no_three_collinear(points):
        return False
    # Check no four concyclic
    if not _no_four_concyclic(points, D):
        return False
    return True

def _format_output(points, D):
    """Format points for output."""
    return {'D': D, 'points': [[[p[0].numerator, p[0].denominator],
                                  [p[1].numerator, p[1].denominator]] for p in points]}

def _fast_dist_check_int(x1, y1, x2, y2, D):
    """
    Fast integer-based check: for integer coordinates, check if
    (x1-x2)^2 + D*(y1-y2)^2 is a perfect square integer.
    Returns True if it's a perfect square.
    """
    dx = x1 - x2
    dy = y1 - y2
    sq = dx * dx + D * dy * dy
    if sq < 0:
        return False
    r = math.isqrt(sq)
    return r * r == sq

def _fast_collinear_int(x1, y1, x2, y2, x3, y3):
    """Fast integer collinearity check. Returns True if collinear."""
    return (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1)

def _fast_concyclic_int(p1, p2, p3, p4, D):
    """
    Fast integer concyclicity check for integer-coordinate points.
    Returns True if concyclic.
    """
    # Use integer arithmetic throughout
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]
    # x^2 + D*y^2 values
    w1 = x1 * x1 + D * y1 * y1
    w2 = x2 * x2 + D * y2 * y2
    w3 = x3 * x3 + D * y3 * y3
    w4 = x4 * x4 + D * y4 * y4
    # 4x4 determinant with integer entries
    m = [[w1, x1, y1, 1],
         [w2, x2, y2, 1],
         [w3, x3, y3, 1],
         [w4, x4, y4, 1]]
    det = (m[0][0] * (m[1][1] * m[2][2] * m[3][3] + m[1][2] * m[2][3] * m[3][1] + m[1][3] * m[2][1] * m[3][2]
                      - m[1][1] * m[2][3] * m[3][2] - m[1][2] * m[2][1] * m[3][3] - m[1][3] * m[2][2] * m[3][1])
           - m[0][1] * (m[1][0] * m[2][2] * m[3][3] + m[1][2] * m[2][3] * m[3][0] + m[1][3] * m[2][0] * m[3][2]
                        - m[1][0] * m[2][3] * m[3][2] - m[1][2] * m[2][0] * m[3][3] - m[1][3] * m[2][2] * m[3][0])
           + m[0][2] * (m[1][0] * m[2][1] * m[3][3] + m[1][1] * m[2][3] * m[3][0] + m[1][3] * m[2][0] * m[3][1]
                        - m[1][0] * m[2][3] * m[3][1] - m[1][1] * m[2][0] * m[3][3] - m[1][3] * m[2][1] * m[3][0])
           - m[0][3] * (m[1][0] * m[2][1] * m[3][2] + m[1][1] * m[2][2] * m[3][0] + m[1][2] * m[2][0] * m[3][1]
                        - m[1][0] * m[2][2] * m[3][1] - m[1][1] * m[2][0] * m[3][2] - m[1][2] * m[2][1] * m[3][0]))
    return det == 0

def construct():
    """
    Search for integral point sets in general position.
    Returns the best (largest) valid set found.
    """
    global _START_TIME
    _START_TIME = time.time()

    # Start with a verified 5-point set for D=1
    # Using the classic 5-point integral set:
    # (0,0), (3,4), (0,4), (3,0), and we need a 5th...
    # Actually let's use a known valid 5-point set.
    # The set {(0,0), (3,4), (4,0), (0,3), (4,3)} - need to verify
    # Let me use a simpler approach: start with 3 known good points and grow.
    
    # Known valid 3-point set: (0,0), (3,4), (0,4)
    # dist(0,0)-(3,4) = 5, dist(0,0)-(0,4) = 4, dist(3,4)-(0,4) = 3. All rational.
    # Not collinear (triangle).
    
    # Let's use a well-known 5-point integral set.
    # One such set: (0,0), (3,4), (0,4), (3,0), (4,3)
    # Check: dist(0,0)-(3,4)=5, dist(0,0)-(0,4)=4, dist(0,0)-(3,0)=3, dist(0,0)-(4,3)=5
    # dist(3,4)-(0,4)=3, dist(3,4)-(3,0)=4, dist(3,4)-(4,3)=sqrt(1+1)=sqrt(2) NOT rational!
    # So that doesn't work.
    
    # Let me try: (0,0), (3,4), (0,4), (3,0) - only 4 points.
    # Add (4,3): dist to (3,4) = sqrt(1+1) = sqrt(2). No.
    # Add (5,0): dist to (3,4) = sqrt(4+16)=sqrt(20). No.
    # Add (0,3): dist to (3,4) = sqrt(9+1)=sqrt(10). No.
    
    # Let me use the set from literature: 
    # A known 5-point integral set in general position:
    # Actually, let me just start with 3 points and grow carefully.
    
    # Start with (0,0), (3,4), (0,4) - verified 3-point set
    initial = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(0), Fraction(4))]
    D = 1
    
    # Verify
    if not _validate(initial, D):
        # Fallback: even simpler
        initial = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(0)), (Fraction(0), Fraction(4))]
        if not _validate(initial, D):
            initial = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))]
    
    best_points = list(initial)
    best_D = D
    best_count = len(best_points)
    
    # Precompute integer representations for fast checking
    # For D=1, points have integer coords
    def get_int_coords(points):
        """Convert Fraction points to integer tuples if possible."""
        result = []
        for p in points:
            x, y = p[0], p[1]
            if x.denominator == 1 and y.denominator == 1:
                result.append((x.numerator, y.numerator))
            else:
                return None  # Not all integer
        return result
    
    # Fast candidate check for integer points with D=1
    def can_add_fast_int(candidate_int, current_int_points, D_val):
        """
        Fast check using integer arithmetic for adding an integer point.
        candidate_int: (x, y) as integers
        current_int_points: list of (x, y) integer tuples
        Returns True if candidate can be added.
        """
        cx, cy = candidate_int
        n = len(current_int_points)
        
        # Check distances to all existing points
        for px, py in current_int_points:
            if not _fast_dist_check_int(cx, cy, px, py, D_val):
                return False
        
        # Check collinearity with all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if _fast_collinear_int(current_int_points[i][0], current_int_points[i][1],
                                       current_int_points[j][0], current_int_points[j][1],
                                       cx, cy):
                    return False
        
        # Check concyclicity with all triples
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if _fast_concyclic_int(current_int_points[i], current_int_points[j],
                                           current_int_points[k], (cx, cy), D_val):
                        return False
        
        return True
    
    # Main search loop: try to grow the set
    # Phase 1: Try to add points with integer coordinates (D=1)
    # Use a structured search with increasing radius
    
    search_radius = 10
    max_radius = 200
    
    while best_count < 12 and _time_left() > 10:
        # Get current integer points
        cur_int = get_int_coords(best_points)
        if cur_int is None:
            break  # Can't do fast search with non-integer points
        
        # Determine search bounds based on current points
        if cur_int:
            x_min = min(p[0] for p in cur_int) - search_radius
            x_max = max(p[0] for p in cur_int) + search_radius
            y_min = min(p[1] for p in cur_int) - search_radius
            y_max = max(p[1] for p in cur_int) + search_radius
        else:
            x_min, x_max = -search_radius, search_radius
            y_min, y_max = -search_radius, search_radius
        
        # Limit the search area
        area = (x_max - x_min + 1) * (y_max - y_min + 1)
        if area > 500000:
            # Too large, reduce radius
            search_radius = max(5, search_radius // 2)
            continue
        
        found = False
        for x in range(x_min, x_max + 1):
            if _time_left() < 5:
                break
            for y in range(y_min, y_max + 1):
                if _time_left() < 3:
                    break
                candidate_int = (x, y)
                # Skip if already in set
                if candidate_int in cur_int:
                    continue
                if can_add_fast_int(candidate_int, cur_int, 1):
                    # Verify with full Fraction-based validation
                    candidate_frac = (Fraction(x), Fraction(y))
                    new_set = best_points + [candidate_frac]
                    if _validate(new_set, 1):
                        best_points = new_set
                        best_count += 1
                        found = True
                        # Reset search radius for next iteration
                        search_radius = 10
                        break
            if found or _time_left() < 5:
                break
        
        if found:
            search_radius = 10
        else:
            # Increase search radius
            search_radius = min(max_radius, search_radius * 2)
            if search_radius >= max_radius:
                # Try a wider random search
                if _time_left() > 20:
                    found_random = False
                    for _ in range(2000):
                        if _time_left() < 5:
                            break
                        rx = random.randint(-300, 300)
                        ry = random.randint(-300, 300)
                        candidate_int = (rx, ry)
                        if candidate_int in cur_int:
                            continue
                        if can_add_fast_int(candidate_int, cur_int, 1):
                            candidate_frac = (Fraction(rx), Fraction(ry))
                            new_set = best_points + [candidate_frac]
                            if _validate(new_set, 1):
                                best_points = new_set
                                best_count += 1
                                found_random = True
                                break
                    if not found_random:
                        break  # Give up on D=1
                else:
                    break
    
    # Phase 2: Try D=2 with rational coordinates
    if _time_left() > 30 and best_count < 12:
        # For D=2, try points with small rational coordinates
        # Start with a small set
        D2 = 2
        # Try (0,0), (1,0), (0,1) with D=2
        # dist((0,0),(1,0)) = 1, dist((0,0),(0,1)) = sqrt(2), dist((1,0),(0,1)) = sqrt(1+2)=sqrt(3)
        # Not all rational. Need to be more careful.
        # For D=2, dist^2 = dx^2 + 2*dy^2 must be a perfect square.
        # (0,0) to (1,1): 1+2=3, not square.
        # (0,0) to (2,1): 4+2=6, not square.
        # (0,0) to (1,2): 1+8=9=3^2. Yes!
        # So (0,0) and (1,2) work with D=2.
        # (0,0) to (2,0): 4, sqrt=2. Yes.
        # (1,2) to (2,0): 1+8=9. Yes!
        # So {(0,0), (1,2), (2,0)} is a valid 3-point set for D=2.
        # Check collinearity: (0,0),(1,2),(2,0): slope 0-1 = 2/1=2, slope 0-2 = 0/2=0. Not collinear.
        
        # Try to add more points for D=2
        init2 = [(Fraction(0), Fraction(0)), (Fraction(1), Fraction(2)), (Fraction(2), Fraction(0))]
        if _validate(init2, D2):
            # Try to grow
            cur_int2 = get_int_coords(init2)
            if cur_int2:
                for x in range(-20, 21):
                    if _time_left() < 5:
                        break
                    for y in range(-20, 21):
                        if _time_left() < 3:
                            break
                        cand = (x, y)
                        if cand in cur_int2:
                            continue
                        if can_add_fast_int(cand, cur_int2, D2):
                            cand_frac = (Fraction(x), Fraction(y))
                            new_set = init2 + [cand_frac]
                            if _validate(new_set, D2):
                                if len(new_set) > best_count:
                                    best_points = new_set
                                    best_D = D2
                                    best_count = len(new_set)
    
    # Phase 3: Try to add points using Fraction coordinates for D=1
    # Search for points with small denominators
    if _time_left() > 20 and best_D == 1:
        cur_int = get_int_coords(best_points)
        if cur_int:
            # Try points with denominator 2, 3, etc.
            for denom in [2, 3, 4, 5, 6, 8, 10, 12]:
                if _time_left() < 5:
                    break
                # Search in a bounded region
                bound = 30
                for x_num in range(-bound * denom, bound * denom + 1, denom):
                    if _time_left() < 3:
                        break
                    for y_num in range(-bound * denom, bound * denom + 1, denom):
                        if _time_left() < 3:
                            break
                        # Only check if not already present
                        cand = (Fraction(x_num, denom), Fraction(y_num, denom))
                        # Check if already in set
                        if cand in best_points:
                            continue
                        # Quick distance check
                        ok = True
                        for p in best_points:
                            if not _dist_is_rational(cand, p, 1):
                                ok = False
                                break
                        if not ok:
                            continue
                        # Check collinearity
                        ok = True
                        for i in range(len(best_points)):
                            for j in range(i + 1, len(best_points)):
                                if _collinear_check(best_points[i][0], best_points[i][1],
                                                    best_points[j][0], best_points[j][1],
                                                    cand[0], cand[1]):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        # Check concyclicity
                        ok = True
                        for combo in combinations(range(len(best_points)), 3):
                            if _concyclic_check(best_points[combo[0]], best_points[combo[1]],
                                                best_points[combo[2]], cand, 1):
                                ok = False
                                break
                        if ok:
                            new_set = best_points + [cand]
                            if _validate(new_set, 1):
                                best_points = new_set
                                best_count += 1
                                break
                    if best_count > len(best_points) - 1:
                        break
    
    return _format_output(best_points, best_D)
