"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def _time_left():
    return _DEADLINE - _elapsed()

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist(p, q, D):
    # p = (x, y) where actual point is (x, y*sqrt(D))
    # distance^2 = (x1-x2)^2 + D*(y1-y2)^2
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + D * dy * dy

def _check_rational_distances(points, D):
    """Check all pairwise distances are rational."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _sqdist(points[i], points[j], D)
            if not _is_square(d2):
                return False
    return True

def _check_no_three_collinear(points, D):
    """No three points collinear. Points are (x, y*sqrt(D))."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Three points (x1,y1), (x2,y2), (x3,y3) are collinear iff
                # (x2-x1)(y3-y1) = (x3-x1)(y2-y1)
                # With y = y'*sqrt(D): (x2-x1)*y3'*sqrt(D) = (x3-x1)*y2'*sqrt(D)
                # => (x2-x1)*y3' = (x3-x1)*y2'
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * y3 == (x3 - x1) * y2:
                    return False
    return True

def _check_no_four_concyclic(points, D):
    """No four points concyclic. Points are (x, y*sqrt(D))."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Four points are concyclic iff the determinant is zero.
                    # For points (xi, yi) the circle equation is:
                    # x^2 + y^2 + Ax + By + C = 0
                    # With y = y'*sqrt(D), we have y^2 = D*(y')^2
                    # So: x^2 + D*(y')^2 + A*x + B*y'*sqrt(D) + C = 0
                    # This must hold for all four points.
                    # Let's use the determinant form.
                    # For point (x, y*sqrt(D)): x^2 + D*y^2 + A*x + B*y*sqrt(D) + C = 0
                    # Separate into rational and sqrt(D) parts:
                    # (x^2 + D*y^2 + A*x + C) + B*y*sqrt(D) = 0
                    # Since sqrt(D) is irrational (D squarefree > 1) or rational (D=1):
                    # If D=1: standard circle check
                    # If D>1: we need both rational and sqrt(D) parts to be zero.
                    #   x^2 + D*y^2 + A*x + C = 0  and  B*y = 0 for all 4 points
                    #   If any y != 0, then B=0, and we're back to x^2 + D*y^2 + A*x + C = 0
                    #   which is an ellipse, not a circle in the Euclidean sense.
                    # 
                    # Actually, "concyclic" means lying on a circle in the Euclidean plane.
                    # The Euclidean coordinates are (x, y*sqrt(D)).
                    # Circle: (X - cx)^2 + (Y - cy)^2 = r^2
                    # X^2 - 2*cx*X + cx^2 + Y^2 - 2*cy*Y + cy^2 = r^2
                    # X^2 + Y^2 - 2*cx*X - 2*cy*Y + (cx^2 + cy^2 - r^2) = 0
                    # Substitute X=x, Y=y*sqrt(D):
                    # x^2 + D*y^2 - 2*cx*x - 2*cy*sqrt(D)*y + K = 0
                    # Group: (x^2 + D*y^2 - 2*cx*x + K) + (-2*cy*sqrt(D))*y = 0
                    # For this to hold for 4 points with different y values, we need
                    # -2*cy*y = 0 for all, so either cy=0 or all y=0.
                    # If cy=0: x^2 + D*y^2 - 2*cx*x + K = 0 for all 4 points.
                    # This is a system of 4 equations in cx, K (2 unknowns) with 4 points.
                    # Generally overdetermined, so no 4 concyclic unless special.
                    
                    # Let me just check: do 4 points lie on some circle?
                    # Use the general circle equation in (X,Y) = (x, y*sqrt(D)):
                    # X^2 + Y^2 + aX + bY + c = 0
                    # For 4 points, solve the 3x3 system for 3 of them, check the 4th.
                    p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                    # X^2 + Y^2 + aX + bY + c = 0
                    # For each point: X^2 + Y^2 + aX + bY + c = 0
                    # Subtract first from others to eliminate c:
                    # (X2^2+Y2^2 - X1^2-Y1^2) + a(X2-X1) + b(Y2-Y1) = 0
                    # (X3^2+Y3^2 - X1^2-Y1^2) + a(X3-X1) + b(Y3-Y1) = 0
                    # Solve 2x2 for a, b, then check point 4.
                    
                    X1, Y1 = p1[0], p1[1] * D  # Wait, Y = y*sqrt(D), but I'm using rational arithmetic
                    # Problem: Y = y*sqrt(D) is not rational if D>1 and y!=0.
                    # I need to work with the actual Euclidean coordinates.
                    # Let me use the form: X^2 + Y^2 + aX + bY + c = 0 where X=x, Y=y*sqrt(D)
                    # X^2 + Y^2 = x^2 + D*y^2 (rational)
                    # aX = a*x (rational)
                    # bY = b*y*sqrt(D) (involves sqrt(D))
                    # c (rational)
                    # So: (x^2 + D*y^2 + a*x + c) + b*y*sqrt(D) = 0
                    # For this to be zero, both parts must be zero (since sqrt(D) irrational for squarefree D>1):
                    # Part 1: x^2 + D*y^2 + a*x + c = 0
                    # Part 2: b*y = 0
                    # For 4 points, if not all y=0, then b=0, and we need:
                    # x^2 + D*y^2 + a*x + c = 0 for all 4 points.
                    # This is 4 equations in a, c (2 unknowns). Overdetermined.
                    # Check if consistent:
                    p1x, p1y = p1[0], p1[1]
                    p2x, p2y = p2[0], p2[1]
                    p3x, p3y = p3[0], p3[1]
                    p4x, p4y = p4[0], p4[1]
                    
                    # If D == 1: standard circle check
                    if D == 1:
                        # X^2+Y^2+aX+bY+c=0 for all 4
                        # Use determinant method
                        # |X^2+Y^2, X, Y, 1| for 4 points, det should be 0
                        v1 = [p1x*p1x + p1y*p1y, p1x, p1y, 1]
                        v2 = [p2x*p2x + p2y*p2y, p2x, p2y, 1]
                        v3 = [p3x*p3x + p3y*p3y, p3x, p3y, 1]
                        v4 = [p4x*p4x + p4y*p4y, p4x, p4y, 1]
                        # Compute 4x4 determinant
                        # Use cofactor expansion
                        def det4(m):
                            # m is 4x4 list of lists
                            d = 0
                            for j in range(4):
                                # Minor
                                minor = []
                                for i in range(1, 4):
                                    row = []
                                    for k in range(4):
                                        if k != j:
                                            row.append(m[i][k])
                                    minor.append(row)
                                # det of 3x3
                                d3 = (minor[0][0]*(minor[1][1]*minor[2][2]-minor[1][2]*minor[2][1])
                                      - minor[0][1]*(minor[1][0]*minor[2][2]-minor[1][2]*minor[2][0])
                                      + minor[0][2]*(minor[1][0]*minor[2][1]-minor[1][1]*minor[2][0]))
                                sign = 1 if j % 2 == 0 else -1
                                d += sign * m[0][j] * d3
                            return d
                        M = [v1, v2, v3, v4]
                        if det4(M) != 0:
                            continue  # Not concyclic
                        else:
                            return False  # Four concyclic found
                    
                    else:
                        # D > 1: need b=0 (if any y!=0), then check x^2+D*y^2+a*x+c=0
                        # Actually, we need to check if there exists a,b,c such that
                        # for all 4 points: (x^2+D*y^2+a*x+c) + b*y*sqrt(D) = 0
                        # This means for each point i: (xi^2+D*yi^2+a*xi+c) = -b*yi*sqrt(D)
                        # Left side is rational, right side is rational*sqrt(D).
                        # So both must be zero: b*yi=0 for all i, and xi^2+D*yi^2+a*xi+c=0 for all i.
                        # If all yi=0: then it's just x^2+a*x+c=0, at most 2 solutions, so 4 points impossible unless same x.
                        # If some yi!=0: b=0, need xi^2+D*yi^2+a*xi+c=0 for all 4.
                        # 4 equations, 2 unknowns (a,c). Check consistency.
                        # From points 1,2,3:
                        # a*p1x + c = -(p1x^2+D*p1y^2)
                        # a*p2x + c = -(p2x^2+D*p2y^2)
                        # a*p3x + c = -(p3x^2+D*p3y^2)
                        # Subtract 1 from 2: a*(p2x-p1x) = -(p2x^2+D*p2y^2 - p1x^2-D*p1y^2)
                        # If p2x==p1x: need p2x^2+D*p2y^2 == p1x^2+D*p1y^2, i.e., D*(p2y^2-p1y^2)=0, so p2y^2=p1y^2
                        # This gets complicated. Let me just check if the 4 points satisfy a common circle
                        # by trying to find a,b,c.
                        
                        # Simplified: if any y_i != 0, b must be 0.
                        # Then check if x^2+D*y^2+a*x+c=0 for all 4.
                        # Use first 3 to solve for a,c, check 4th.
                        all_y_zero = (p1y == 0 and p2y == 0 and p3y == 0 and p4y == 0)
                        if all_y_zero:
                            # Circle in x-axis: at most 2 distinct x values can satisfy x^2+ax+c=0
                            # So 4 distinct points on x-axis can't be concyclic (they'd need to be on a circle,
                            # but a circle intersects x-axis in at most 2 points)
                            continue
                        else:
                            # b=0, solve for a,c from points 1,2
                            # a*p1x + c = -V1 where V1 = p1x^2+D*p1y^2
                            # a*p2x + c = -V2
                            V1 = p1x*p1x + D*p1y*p1y
                            V2 = p2x*p2x + D*p2y*p2y
                            V3 = p3x*p3x + D*p3y*p3y
                            V4 = p4x*p4x + D*p4y*p4y
                            dx12 = p2x - p1x
                            if dx12 == 0:
                                # Can't solve from 1,2. Try 1,3
                                dx13 = p3x - p1x
                                if dx13 == 0:
                                    # All same x? Then not distinct enough or collinear
                                    continue
                                a_num = V3 - V1
                                a_den = dx13
                                if a_den != 0 and a_num % a_den == 0:
                                    a_val = a_num // a_den
                                    c_val = -V1 - a_val * p1x
                                    # Check point 2 and 4
                                    if p2x*p2x + D*p2y*p2y + a_val*p2x + c_val != 0:
                                        continue
                                    if p4x*p4x + D*p4y*p4y + a_val*p4x + c_val != 0:
                                        continue
                                    return False
                                else:
                                    continue
                            else:
                                a_num = V2 - V1
                                a_den = dx12
                                if a_den != 0 and a_num % a_den == 0:
                                    a_val = a_num // a_den
                                    c_val = -V1 - a_val * p1x
                                    if p3x*p3x + D*p3y*p3y + a_val*p3x + c_val != 0:
                                        continue
                                    if p4x*p4x + D*p4y*p4y + a_val*p4x + c_val != 0:
                                        continue
                                    return False
                                else:
                                    continue
    return True

def _validate(points, D):
    """Full validation."""
    n = len(points)
    if n < 3:
        return False
    # Check distinctness
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
    # Check rational distances
    if not _check_rational_distances(points, D):
        return False
    # Check no three collinear
    if not _check_no_three_collinear(points, D):
        return False
    # Check no four concyclic
    if not _check_no_four_concyclic(points, D):
        return False
    return True

def _try_add_point(points, D, candidate):
    """Try to add candidate to points, return True if valid."""
    new_points = points + [candidate]
    return _validate(new_points, D)

def construct():
    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    
    # Verify base
    if not _validate(base_points, D):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in base_points[:3]]}
    
    best = list(base_points)
    
    # Try to add more points by searching over a grid
    # For D=1, points are just integer lattice points
    # We need integer points with integer pairwise distances (Eisenstein-like / integral point sets)
    
    # Search strategy: try points on a larger grid
    # Known integral point sets in 2D: max known is 7 (I think), but let's search for 6, 7
    
    # Try adding points one at a time
    search_range = 300
    max_points_found = len(best)
    
    # Generate candidate points
    # To keep it fast, try points that are at integer distance from all existing points
    for attempt in range(50):
        if _time_left() < 10:
            break
        if len(best) >= 8:
            break
            
        added = False
        for x in range(-search_range, search_range + 1):
            if _time_left() < 10:
                break
            for y in range(-search_range, search_range + 1):
                if _time_left() < 5:
                    break
                candidate = (x, y)
                # Quick check: integer distance to all existing points
                ok = True
                for p in best:
                    d2 = (x - p[0])**2 + (y - p[1])**2
                    if not _is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                # Full validation
                if _try_add_point(best, D, candidate):
                    best.append(candidate)
                    added = True
                    max_points_found = len(best)
                    if len(best) >= 8:
                        break
            if len(best) >= 8:
                break
        if not added:
            # Try a different starting configuration or expand search
            # Try shifting the base configuration
            break
    
    # If we couldn't add more, try a different approach:
    # Search for 6-point integral point sets directly
    if len(best) < 6 and _time_left() > 30:
        # Try known configurations or search differently
        # For now, return what we have
        pass
    
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
