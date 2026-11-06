"""Find maximal integral point sets in general position."""
import time
from itertools import combinations

_START = None

def _elapsed():
    global _START
    if _START is None:
        import time as _t
        _START = _t.time()
    return time.time() - _START

def _deadline_check(limit=130.0):
    return _elapsed() < limit

def _sq(n):
    return n * n

def _is_square(n):
    if n < 0:
        return False
    r = int(n ** 0.5)
    while r * r < n:
        r += 1
    while r * r > n:
        r -= 1
    return r * r == n

def _dist_sq(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a1 = x1 - x2
    b1 = y1 - y2
    c1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
    
    a2 = x1 - x3
    b2 = y1 - y3
    c2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return False
    
    D_num = c1 * b2 - c2 * b1
    E_num = a1 * c2 - a2 * c1
    
    return (det * (x4*x4 + y4*y4) + D_num * x4 + E_num * y4 
            - det * (x1*x1 + y1*y1) - D_num * x1 - E_num * y1) == 0

def _valid_set(points):
    """Check all constraints for a set of points."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _dist_sq(points[i], points[j])
            if dsq == 0:
                return False
            if not _is_square(dsq):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _find_integer_distance_points(existing, max_coord=200):
    """Find all integer points at integer distance from all existing points."""
    candidates = []
    n = len(existing)
    if n == 0:
        return candidates
    
    # For each existing point, the new point must be on a circle of integer radius
    # Intersect circles from at least 2 points to get candidate locations
    # Use the first two points to generate candidates
    
    x0, y0 = existing[0]
    x1, y1 = existing[1]
    dx = x1 - x0
    dy = y1 - y0
    d2 = dx * dx + dy * dy
    
    if d2 == 0:
        return candidates
    
    # For integer distances r0, r1 from points 0 and 1:
    # (x-x0)^2 + (y-y0)^2 = r0^2
    # (x-x1)^2 + (y-y1)^2 = r1^2
    # Subtract: 2*dx*(x-x0) + 2*dy*(y-y0) = r0^2 - r1^2 - d2... 
    # Actually: (x-x1)^2-(x-x0)^2 + (y-y1)^2-(y-y0)^2 = r1^2 - r0^2
    # -2*dx*x + dx^2 - 2*dy*y + dy^2 = r1^2 - r0^2
    # 2*dx*x + 2*dy*y = r0^2 - r1^2 + d2
    # dx*x + dy*y = (r0^2 - r1^2 + d2) / 2
    
    # For this to have integer solutions, (r0^2 - r1^2 + d2) must be even
    # and we need to find integer (x,y) on this line at distance r0 from (x0,y0)
    
    for r0 in range(1, max_coord + 1):
        if not _deadline_check(120.0):
            break
        for r1 in range(1, max_coord + 1):
            if not _deadline_check(120.0):
                break
            rhs = r0 * r0 - r1 * r1 + d2
            if rhs % 2 != 0:
                continue
            rhs_half = rhs // 2
            # Line: dx*x + dy*y = rhs_half
            # Need gcd(dx,dy) | rhs_half for integer solutions
            g = abs(dx)
            h = abs(dy)
            while h:
                g, h = h, g % h
            g = g if g != 0 else 1
            if rhs_half % g != 0:
                continue
            
            # Find particular solution
            # dx*x + dy*y = rhs_half
            # Use extended Euclidean
            def _egcd(a, b):
                if b == 0:
                    return (1, 0, a)
                x, y, g = _egcd(b, a % b)
                return (y, x - (a // b) * y, g)
            
            if dx != 0 or dy != 0:
                xx, yy, gg = _egcd(dx, dy)
                # dx*xx + dy*yy = gg
                # Multiply by rhs_half/gg
                factor = rhs_half // gg
                x_part = xx * factor
                y_part = yy * factor
                # General solution: x = x_part + (dy/g)*t, y = y_part - (dx/g)*t
                dx_g = dx // gg
                dy_g = dy // gg
                # Check if (x_part + dy_g*t, y_part - dx_g*t) is at distance r0 from (x0,y0)
                # (x_part + dy_g*t - x0)^2 + (y_part - dx_g*t - y0)^2 = r0^2
                # This is a quadratic in t
                ax = x_part - x0
                ay = y_part - y0
                # (ax + dy_g*t)^2 + (ay - dx_g*t)^2 = r0^2
                # (dy_g^2 + dx_g^2)*t^2 + 2*(ax*dy_g - ay*dx_g)*t + (ax^2 + ay^2 - r0^2) = 0
                A = dy_g * dy_g + dx_g * dx_g
                B = 2 * (ax * dy_g - ay * dx_g)
                C = ax * ax + ay * ay - r0 * r0
                if A == 0:
                    continue
                disc = B * B - 4 * A * C
                if disc < 0:
                    continue
                if not _is_square(disc):
                    continue
                sqrt_disc = int(disc ** 0.5)
                # t = (-B ± sqrt_disc) / (2*A)
                for sign in [1, -1]:
                    num = -B + sign * sqrt_disc
                    if num % (2 * A) != 0:
                        continue
                    t = num // (2 * A)
                    x_new = x_part + dy_g * t
                    y_new = y_part - dx_g * t
                    # Verify it's at integer distance from ALL existing points
                    valid = True
                    for ep in existing:
                        dsq = (x_new - ep[0]) ** 2 + (y_new - ep[1]) ** 2
                        if dsq == 0 or not _is_square(dsq):
                            valid = False
                            break
                    if valid:
                        # Check not collinear with any pair
                        collinear = False
                        for i in range(len(existing)):
                            for j in range(i + 1, len(existing)):
                                if _are_collinear(existing[i], existing[j], (x_new, y_new)):
                                    collinear = True
                                    break
                            if collinear:
                                break
                        if not collinear:
                            # Check not concyclic with any triple
                            concyclic = False
                            for i in range(len(existing)):
                                for j in range(i + 1, len(existing)):
                                    for k in range(j + 1, len(existing)):
                                        if _are_concyclic(existing[i], existing[j], existing[k], (x_new, y_new)):
                                            concyclic = True
                                            break
                                    if concyclic:
                                        break
                                if concyclic:
                                    break
                            if not concyclic:
                                candidates.append((x_new, y_new))
    
    # Deduplicate
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def construct():
    global _START
    _START = time.time()
    deadline = 125.0
    
    # Known 5-point solution as fallback
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = fallback
    best_size = 5
    
    # Try to find a 6th point using circle intersection method
    if _deadline_check(deadline):
        candidates6 = _find_integer_distance_points(fallback, max_coord=150)
        if candidates6:
            best = fallback + [candidates6[0]]
            best_size = 6
    
    # If we found 6, try to find 7
    if best_size == 6 and _deadline_check(deadline):
        candidates7 = _find_integer_distance_points(best, max_coord=120)
        if candidates7:
            best = best + [candidates7[0]]
            best_size = 7
    
    # If we found 7, try to find 8
    if best_size == 7 and _deadline_check(deadline):
        candidates8 = _find_integer_distance_points(best, max_coord=100)
        if candidates8:
            best = best + [candidates8[0]]
            best_size = 8
    
    # Format result
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': D, 'points': points_out}
