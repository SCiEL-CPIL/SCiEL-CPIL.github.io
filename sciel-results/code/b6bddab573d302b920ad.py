"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from fractions import Fraction

_START_TIME = time.time()
_TIMEOUT = 110.0  # safety margin before 120s wall-clock limit

def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _check_distances(points):
    """Check all pairwise distances are rational (integers for D=1)."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + dy * dy
            if not _is_square(d2):
                return False
    return True

def _check_collinear(points):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                # Cross product
                if (xj - xi) * (yk - yi) - (yj - yi) * (xk - xi) == 0:
                    return False
    return True

def _check_concyclic(points):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                # Find circle through i, j, k
                # (x - a)^2 + (y - b)^2 = r^2
                # x^2 - 2ax + a^2 + y^2 - 2by + b^2 = r^2
                # x^2 + y^2 - 2ax - 2by + (a^2 + b^2 - r^2) = 0
                # Solve for a, b, c where c = a^2+b^2-r^2
                # x^2 + y^2 + 2ax + 2by + c = 0 (sign flip)
                # Let's use: A*x + B*y + C = x^2 + y^2
                # For point (x,y): A*x + B*y + C = x^2 + y^2
                # We have 3 equations, 3 unknowns (A, B, C)
                # Solve the linear system
                # x1*A + y1*B + C = x1^2 + y1^2
                # x2*A + y2*B + C = x2^2 + y2^2
                # x3*A + y3*B + C = x3^2 + y3^2
                # Subtract eq1 from eq2 and eq3:
                # (x2-x1)*A + (y2-y1)*B = x2^2-x1^2 + y2^2-y1^2
                # (x3-x1)*A + (y3-y1)*B = x3^2-x1^2 + y3^2-y1^2
                dx2 = xj - xi
                dy2 = yj - yi
                rhs2 = xj*xj + yj*yj - xi*xi - yi*yi
                dx3 = xk - xi
                dy3 = yk - yi
                rhs3 = xk*xk + yk*yk - xi*xi - yi*yi
                det = dx2 * dy3 - dx3 * dy2
                if det == 0:
                    continue  # collinear, skip (already checked)
                # A = (rhs2*dy3 - rhs3*dy2) / det
                # B = (dx2*rhs3 - dx3*rhs2) / det
                # C = xi^2 + yi^2 - A*xi - B*yi
                # Check if point (xl, yl) satisfies: xl^2+yl^2 = A*xl + B*yl + C
                # Multiply through by det to avoid fractions:
                # det*(xl^2+yl^2) = (rhs2*dy3 - rhs3*dy2)*xl + (dx2*rhs3 - dx3*rhs2)*yl + C_det
                # where C_det = det*(xi^2+yi^2) - A_num*xi - B_num*yi
                A_num = rhs2 * dy3 - rhs3 * dy2
                B_num = dx2 * rhs3 - dx3 * rhs2
                C_num = det * (xi*xi + yi*yi) - A_num * xi - B_num * yi
                for l in range(k + 1, n):
                    xl, yl = points[l]
                    lhs = det * (xl*xl + yl*yl)
                    rhs = A_num * xl + B_num * yl + C_num
                    if lhs == rhs:
                        return False
    return True

def _is_valid(points):
    if len(points) < 3:
        return False
    if not _check_distances(points):
        return False
    if not _check_collinear(points):
        return False
    if not _check_concyclic(points):
        return False
    return True

def _format_points(points):
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}

def _local_search(points, max_steps=5000):
    """Try to add one more point by searching lattice positions."""
    n = len(points)
    # Determine search range based on existing points
    max_x = max(abs(p[0]) for p in points) + 20
    max_y = max(abs(p[1]) for p in points) + 20
    
    # Generate candidate points
    # First try points near existing ones, then expand
    candidates = set()
    for x in range(-max_x, max_x + 1):
        for y in range(-max_y, max_y + 1):
            if (x, y) not in points:
                candidates.add((x, y))
    
    # Sort candidates by distance to centroid (prefer close points first)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    candidates = sorted(candidates, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
    
    for cand in candidates:
        if _time_left() < 2.0:
            break
        new_points = points + [cand]
        if _is_valid(new_points):
            return new_points
    return None

def construct():
    # Start with known 5-point configuration
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to extend to 6, 7, 8, ... points
    current = points
    for target in range(6, 13):
        if _time_left() < 5.0:
            break
        result = _local_search(current, max_steps=10000)
        if result is not None:
            current = result
        else:
            # Try a different starting configuration or random restart
            break
    
    return _format_points(current)
# EVOLVE-BLOCK-END
