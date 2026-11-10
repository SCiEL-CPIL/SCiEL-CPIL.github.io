"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import math
import time

# EVOLVE-BLOCK-START

def _is_square(n):
    """Check if n is a perfect square (n >= 0)."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _rational_sqrt(n):
    """Return Fraction(sqrt(n)) if n is a perfect square, else None."""
    if _is_square(n):
        return Fraction(math.isqrt(n), 1)
    return None

def _dist_sq_rational(p1, p2, D):
    """Compute squared distance between two points (x1,y1*sqrt(D)), (x2,y2*sqrt(D)).
    Returns (num, den) as a reduced fraction, or None if not rational.
    """
    # dist^2 = (x1-x2)^2 + D*(y1-y2)^2
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    # dx^2 is a rational square
    val = dx * dx + D * dy * dy
    return val  # Fraction

def _all_dist_rational(points, D):
    """Check all pairwise squared distances are rational (they always are as Fractions).
    The real constraint is that distances themselves are rational, i.e., dist^2 is a perfect square of a rational.
    """
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist_sq_rational(points[i], points[j], D)
            if d2 < 0:
                return False
            # Check if d2 is a perfect square in Q
            if not _rational_sqrt(d2.numerator) or not _rational_sqrt(d2.denominator):
                return False
            # Also check the ratio is a perfect square
            sr = _rational_sqrt(d2.numerator)
            sd = _rational_sqrt(d2.denominator)
            if sr is None or sd is None:
                return False
    return True

def _check_no_three_collinear(points):
    """Check no three points are collinear using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Cross product (p_j - p_i) x (p_k - p_i)
                # For points in Q x Q*sqrt(D), the cross product involves sqrt(D) terms
                # Actually, collinearity in the embedding space: 
                # Points are (x_i, y_i * sqrt(D)). Three are collinear iff
                # (x_j - x_i)/(x_k - x_i) = (y_j - y_i)/(y_k - y_i) when denominators non-zero
                # More robustly: (x_j-x_i)*(y_k-y_i) - (x_k-x_i)*(y_j-y_i) = 0
                a = points[j][0] - points[i][0]
                b = points[k][0] - points[i][0]
                c = points[j][1] - points[i][1]
                d = points[k][1] - points[i][1]
                # a*d - b*c = 0
                if a * d == b * c:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """Check no four points are concyclic using exact rational arithmetic."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points are concyclic iff the circle through 3 of them passes through the 4th.
                    # We can check the determinant condition or use the fact that 
                    # for points (x_a, y_a*sqrt(D)), (x_b, y_b*sqrt(D)), etc.
                    # The circle equation in the embedded space: x^2 + y^2*D + Ax + By*sqrt(D)*D + C = 0
                    # Wait, let's think more carefully.
                    # In the embedded Euclidean plane, points are P_i = (x_i, y_i * sqrt(D)).
                    # A circle has equation x^2 + z^2 + ax + bz + c = 0 where z = y*sqrt(D).
                    # So x_i^2 + y_i^2 * D + a*x_i + b*y_i*sqrt(D) + c = 0 for each point on the circle.
                    # For this to hold for rational x_i, y_i and the sqrt(D) term, we need:
                    # x_i^2 + y_i^2*D + a*x_i + c + b*y_i*sqrt(D) = 0
                    # This requires b*y_i = 0 for all i (since sqrt(D) is irrational if D not a square).
                    # So either b=0 (circle center on x-axis) or all y_i=0.
                    # Wait, that's not right. The circle equation is x^2 + z^2 + ax + bz + c = 0.
                    # z = y*sqrt(D), so z^2 = y^2*D, bz = b*y*sqrt(D).
                    # For the equation to hold for rational x, y and irrational sqrt(D) (D squarefree > 1):
                    # x^2 + y^2*D + ax + c + (by)*sqrt(D) = 0
                    # This implies by = 0 for each point on the circle.
                    # So either b = 0 or y_i = 0 for all four points.
                    # If b = 0, the circle is x^2 + y^2*D + ax + c = 0, i.e., (x + a/2)^2 + y^2*D = r^2.
                    # This is an ellipse in the (x,y) plane, a circle in the (x, z=y*sqrt(D)) plane.
                    # 
                    # Actually, let me just use the determinant method. Four points P1,P2,P3,P4 are concyclic iff:
                    # det([[x1^2+y1^2*D, x1, y1*sqrt(D), 1],
                    #      [x2^2+y2^2*D, x2, y2*sqrt(D), 1],
                    #      [x3^2+y3^2*D, x3, y3*sqrt(D), 1],
                    #      [x4^2+y4^2*D, x4, y4*sqrt(D), 1]]) = 0
                    # 
                    # This determinant involves sqrt(D). Let's expand it.
                    # Let z_i = y_i * sqrt(D). Then we need det of matrix with entries:
                    # x_i^2 + z_i^2, x_i, z_i, 1
                    # = x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1
                    # 
                    # The determinant is a linear combination of 1 and sqrt(D) (since each z_i contributes one sqrt(D)).
                    # For the determinant to be zero, both the rational and irrational parts must be zero.
                    # 
                    # Let me compute this more carefully. I'll use the fact that we can separate sqrt(D).
                    # Let's define the matrix M where M[i] = [x_i^2 + D*y_i^2, x_i, y_i*sqrt(D), 1]
                    # We can write y_i*sqrt(D) = y_i + y_i*(sqrt(D)-1) ... no, that's not helpful.
                    # 
                    # Better: let's just compute the determinant symbolically.
                    # Actually, the simplest approach: three non-collinear points define a unique circle.
                    # The circle through P1, P2, P3 has equation x^2 + z^2 + Ax + Bz + C = 0.
                    # Substituting z_i = y_i*sqrt(D):
                    # x_i^2 + D*y_i^2 + A*x_i + B*y_i*sqrt(D) + C = 0
                    # For i=1,2,3, this gives a system. Since sqrt(D) is irrational (D>1 squarefree),
                    # we need B*y_i = 0 for each i. So either B=0 or y_i=0.
                    # 
                    # Case 1: B = 0. Circle is x^2 + D*y^2 + Ax + C = 0.
                    #   Then P4 is on it iff x4^2 + D*y4^2 + A*x4 + C = 0.
                    # Case 2: y1=y2=y3=0. Then P1,P2,P3 are on the x-axis, which means they're collinear
                    #   (already checked), so this case is excluded by the no-three-collinear check.
                    # 
                    # So for D > 1, four points are concyclic iff:
                    #   There exist A, C such that x_i^2 + D*y_i^2 + A*x_i + C = 0 for i=1,2,3,4.
                    # This is a linear system in A, C for each pair, and we need consistency.
                    # 
                    # From points 1 and 2: A = -(x1^2 + D*y1^2 - x2^2 - D*y2^2)/(x1 - x2) (if x1 != x2)
                    # Then check if point 3 and 4 satisfy the same circle.
                    # 
                    # More robustly: solve for A, C using points 1,2,3 and check point 4.
                    # From i=1,2: 
                    #   (x1^2 + D*y1^2) + A*x1 + C = 0
                    #   (x2^2 + D*y2^2) + A*x2 + C = 0
                    # Subtract: (x1^2 - x2^2) + D*(y1^2 - y2^2) + A*(x1-x2) = 0
                    # If x1 != x2: A = -((x1^2 - x2^2) + D*(y1^2 - y2^2))/(x1-x2)
                    # If x1 == x2: need y1^2 != y2^2 for unique solution... actually if x1=x2 and y1^2=y2^2, 
                    #   the points are either identical or symmetric, which is degenerate.
                    # 
                    # Let me just use: for each triple, solve for A,C and check the fourth.
                    # But we need to handle the case where x_i are equal.
                    # 
                    # Alternative: use the determinant of a 4x4 matrix after eliminating sqrt(D).
                    # Since the concyclic condition for D>1 reduces to x_i^2 + D*y_i^2 + A*x_i + C = 0,
                    # we can check if the four points lie on such a curve.
                    # This is equivalent to: the 4x4 matrix
                    # [[x1^2 + D*y1^2, x1, 1],
                    #  [x2^2 + D*y2^2, x2, 1],
                    #  [x3^2 + D*y3^2, x3, 1],
                    #  [x4^2 + D*y4^2, x4, 1]]
                    # having a non-trivial null space... no, we need A*x_i + C = -(x_i^2 + D*y_i^2) for all i.
                    # This is a system of 4 equations in 2 unknowns (A, C). It has a solution iff
                    # the vectors (x_i^2 + D*y_i^2, x_i, 1) for i=1..4 are linearly dependent in a specific way.
                    # 
                    # Simpler: pick 3 points to determine A and C, then check the 4th.
                    # If any two of the 3 have the same x, use a different pair.
                    
                    # Let's just check: for the 4 points, verify that all 4 triples give consistent circles.
                    # Actually, simplest: check if the circle through P_i, P_j, P_k (in the x^2+Dy^2 plane) passes through P_l.
                    
                    # For a set of 4 points, pick 3 that are not degenerate (not all same x).
                    # Compute the circle through those 3 in the "flattened" sense, check the 4th.
                    
                    # Let me just do it directly.
                    pts = [points[i], points[j], points[k], points[l]]
                    # Try to find A, C such that x^2 + D*y^2 + A*x + C = 0 for all 4.
                    # Use first 3 points to solve.
                    # Eq1: x1^2 + D*y1^2 + A*x1 + C = 0
                    # Eq2: x2^2 + D*y2^2 + A*x2 + C = 0
                    # Eq3: x3^2 + D*y3^2 + A*x3 + C = 0
                    # 
                    # From Eq1-Eq2: (x1^2-x2^2) + D*(y1^2-y2^2) + A*(x1-x2) = 0
                    # From Eq1-Eq3: (x1^2-x3^2) + D*(y1^2-y3^2) + A*(x1-x3) = 0
                    # 
                    x1, y1 = pts[0]
                    x2, y2 = pts[1]
                    x3, y3 = pts[2]
                    x4, y4 = pts[3]
                    
                    v1 = x1*x1 + D*y1*y1
                    v2 = x2*x2 + D*y2*y2
                    v3 = x3*x3 + D*y3*y3
                    v4 = x4*x4 + D*y4*y4
                    
                    # Solve for A, C using points 1 and 2 (if x1 != x2)
                    if x1 != x2:
                        A = -(v1 - v2) / (x1 - x2)
                        C = -(v1 + A*x1)
                        # Check point 3
                        if v3 + A*x3 + C != 0:
                            # Not concyclic (in the D>1 sense)
                            continue
                        # Check point 4
                        if v4 + A*x4 + C == 0:
                            return False
                    else:
                        # x1 == x2, try points 1 and 3
                        if x1 != x3:
                            A = -(v1 - v3) / (x1 - x3)
                            C = -(v1 + A*x1)
                            if v2 + A*x2 + C != 0:
                                continue
                            if v4 + A*x4 + C == 0:
                                return False
                        else:
                            # All three have same x, they might be collinear (already checked) 
                            # or form a vertical line. Check if all four satisfy a circle.
                            # If x1=x2=x3, for them to be on a circle x^2+Dy^2+Ax+C=0, we need
                            # x1^2+D*y_i^2 + A*x1 + C = 0 for i=1,2,3, so D*y_i^2 must be constant.
                            # This means y_1^2 = y_2^2 = y_3^2, so they're at most 2 distinct points (symmetric).
                            # Since points are distinct, this is degenerate. Skip.
                            continue
    return True

def _is_valid(points, D):
    """Check all conditions."""
    if not _all_dist_rational(points, D):
        return False
    if not _check_no_three_collinear(points):
        return False
    if D > 1:
        if not _check_no_four_concyclic(points, D):
            return False
    else:
        # D=1, standard concyclic check
        # Four points concyclic iff determinant = 0
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        # det of [[x1^2+y1^2, x1, y1, 1], ...]
                        # Use the formula: four points are concyclic iff
                        # (x1^2+y1^2)(y2-y3) + (x2^2+y2^2)(y3-y1) + (x3^2+y3^2)(y1-y2)
                        # + x1(y2*y3-y3*y2) + x2(y3*y1-y1*y3) + x3(y1*y2-y2*y1) = 0
                        # Wait, let me use the standard determinant.
                        # Actually, for D=1, we can just check if the circle through 3 passes through the 4th.
                        # Circle through P1,P2,P3: x^2+y^2+Ax+By+C=0
                        # Solve for A,B,C and check P4.
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        
                        v1 = x1*x1 + y1*y1
                        v2 = x2*x2 + y2*y2
                        v3 = x3*x3 + y3*y3
                        v4 = x4*x4 + y4*y4
                        
                        # Solve using Cramer's rule or elimination
                        # Eq1: x1*A + y1*B + C = -v1
                        # Eq2: x2*A + y2*B + C = -v2
                        # Eq3: x3*A + y3*B + C = -v3
                        
                        # Eq1-Eq2: (x1-x2)*A + (y1-y2)*B = -(v1-v2)
                        # Eq1-Eq3: (x1-x3)*A + (y1-y3)*B = -(v1-v3)
                        
                        det = (x1-x2)*(y1-y3) - (x1-x3)*(y1-y2)
                        if det == 0:
                            continue  # Degenerate
                        A = (-(v1-v2)*(y1-y3) + (v1-v3)*(y1-y2)) / det
                        B = (-(v1-v2)*(x1-x3) + (v1-v3)*(x1-x2)) / det
                        C = -(v1 + x1*A + y1*B)
                        
                        if x4*A + y4*B + C + v4 == 0:
                            return False
    return True

def _search_6_points():
    """Try to find a 6-point configuration with D=1."""
    # Start with the known 5-point configuration
    base = [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    
    # Try adding a 6th point
    # Search in a grid around the existing points
    # The bounding box is roughly x in [-60, 20], y in [0, 70]
    
    start_time = time.time()
    deadline = start_time + 100  # Leave margin
    
    # Try various D values
    for D in [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17]:
        if time.time() > deadline:
            break
        # For D=1, use the base and try to add points
        if D == 1:
            pts = base[:]
            # Try adding a 6th point from a grid
            found = False
            for x_num in range(-80, 40):
                if found:
                    break
                if time.time() > deadline:
                    break
                for y_num in range(-10, 90):
                    if found:
                        break
                    if time.time() > deadline:
                        break
                    new_pt = (Fraction(x_num), Fraction(y_num))
                    # Check distinctness
                    if any(new_pt == p for p in pts):
                        continue
                    # Check distances to all existing points
                    ok = True
                    for p in pts:
                        dx = new_pt[0] - p[0]
                        dy = new_pt[1] - p[1]
                        d2 = dx*dx + dy*dy
                        if not _is_square(d2.numerator) or not _is_square(d2.denominator):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Check no three collinear with new point
                    n = len(pts)
                    collinear = False
                    for i in range(n):
                        for j in range(i+1, n):
                            a = pts[j][0] - pts[i][0]
                            b = new_pt[0] - pts[i][0]
                            c = pts[j][1] - pts[i][1]
                            d = new_pt[1] - pts[i][1]
                            if a*d == b*c:
                                collinear = True
                                break
                        if collinear:
                            break
                    if collinear:
                        continue
                    # Check no four concyclic
                    pts_new = pts + [new_pt]
                    if _check_no_four_concyclic(pts_new, 1):
                        return {'D': 1, 'points': [[[p[0].numerator, p[0].denominator], [p[1].numerator, p[1].denominator]] for p in pts_new]}
                    found = True  # Don't break, try next x
            if found:
                continue
        
        # For D > 1, try a different search
        # This is complex, so for now just return the 5-point config if no 6-point found
        pass
    
    # Return the 5-point configuration as fallback
    return {'D': 1, 'points': [[[p[0].numerator, p[0].denominator], [p[1].numerator, p[1].denominator]] for p in base]}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for the evaluator's 120s limit
    
    try:
        # Try to find a 6-point configuration
        result = _search_6_points()
        return result
    except Exception as e:
        # Fallback to the known 5-point configuration
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

# EVOLVE-BLOCK-END
