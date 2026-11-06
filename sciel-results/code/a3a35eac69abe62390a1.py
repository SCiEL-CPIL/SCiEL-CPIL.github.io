"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _gcd_num(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a if a != 0 else 1

def _check_pairwise_rational(p1, p2, D):
    """Check if distance between p1 and p2 is rational."""
    x1n, x1d = p1[0]
    y1n, y1d = p1[1]
    x2n, x2d = p2[0]
    y2n, y2d = p2[1]
    
    dx_num = x1n * x2d - x2n * x1d
    dx_den = x1d * x2d
    
    dy_num = y1n * y2d - y2n * y1d
    dy_den = y1d * y2d
    
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    
    if den == 0:
        return False
    if num < 0:
        return False
    
    g = _gcd_num(num, den)
    num_reduced = num // g
    den_reduced = den // g
    
    sqrt_num = int(math.isqrt(num_reduced))
    sqrt_den = int(math.isqrt(den_reduced))
    
    if sqrt_num * sqrt_num != num_reduced:
        return False
    if sqrt_den * sqrt_den != den_reduced:
        return False
    if sqrt_den == 0:
        return False
        
    return True

def _no_three_collinear(points, D):
    """Check no three points are collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi_n, xi_d = points[i][0]
                yi_n, yi_d = points[i][1]
                xj_n, xj_d = points[j][0]
                yj_n, yj_d = points[j][1]
                xk_n, xk_d = points[k][0]
                yk_n, yk_d = points[k][1]
                
                dx_ji_num = xj_n * xi_d - xi_n * xj_d
                dx_ji_den = xj_d * xi_d
                dy_ki_num = yk_n * yi_d - yi_n * yk_d
                dy_ki_den = yk_d * yi_d
                dx_ki_num = xk_n * xi_d - xi_n * xk_d
                dx_ki_den = xk_d * xi_d
                dy_ji_num = yj_n * yi_d - yi_n * yj_d
                dy_ji_den = yj_d * yi_d
                
                cross_num2 = dx_ji_num * dy_ki_num * dx_ki_den * dy_ji_den - dx_ki_num * dy_ji_num * dx_ji_den * dy_ki_den
                cross_den2 = dx_ji_den * dy_ki_den * dx_ki_den * dy_ji_den
                
                if cross_den2 == 0:
                    continue
                if cross_num2 != 0:
                    continue
                return False
    return True

def _no_four_concyclic(points, D):
    """Check no four points are concyclic."""
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    xi_n, xi_d = points[i][0]
                    yi_n, yi_d = points[i][1]
                    xj_n, xj_d = points[j][0]
                    yj_n, yj_d = points[j][1]
                    xk_n, xk_d = points[k][0]
                    yk_n, yk_d = points[k][1]
                    xl_n, xl_d = points[l][0]
                    yl_n, yl_d = points[l][1]
                    
                    xi = Fraction(xi_n, xi_d)
                    yi = Fraction(yi_n, yi_d)
                    xj = Fraction(xj_n, xj_d)
                    yj = Fraction(yj_n, yj_d)
                    xk = Fraction(xk_n, xk_d)
                    yk = Fraction(yk_n, yk_d)
                    xl = Fraction(xl_n, xl_d)
                    yl = Fraction(yl_n, yl_d)
                    
                    A1 = xj - xi
                    B1 = yj - yi
                    C1 = -(xj*xj + D*yj*yj) + (xi*xi + D*yi*yi)
                    
                    A2 = xk - xi
                    B2 = yk - yi
                    C2 = -(xk*xk + D*yk*yk) + (xi*xi + D*yi*yi)
                    
                    det = A1*B2 - A2*B1
                    if det == 0:
                        continue
                    
                    a = (C1*B2 - C2*B1) / det
                    b_prime = (A1*C2 - A2*C1) / det
                    c = -(xi*xi + D*yi*yi) - a*xi - b_prime*yi
                    
                    if xl*xl + D*yl*yl + a*xl + b_prime*yl + c != 0:
                        continue
                    return False
    return True

def _validate(points, D):
    """Validate the full configuration."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    for i in range(n):
        for j in range(i+1, n):
            if not _check_pairwise_rational(points[i], points[j], D):
                return False
    if not _no_three_collinear(points, D):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _find_6_point_set(D=1, search_limit=500, time_limit=30.0):
    """Search for a 6-point set on two lines y=0 and y=h."""
    start = time.time()
    
    # Try various heights h
    for h in range(1, 100):
        if time.time() > start + time_limit:
            break
        
        # Find all x-differences such that dx^2 + h^2 is a perfect square
        valid_dx = set()
        for k in range(h+1, h + 2000):
            dx2 = k*k - h*h
            if dx2 < 0:
                break
            dx = int(math.isqrt(dx2))
            if dx*dx == dx2:
                valid_dx.add(dx)
                if dx != 0:
                    valid_dx.add(-dx)
        
        if not valid_dx:
            continue
        
        # We want 3 points on y=0 and 3 on y=h
        # Fix bottom point at x=0
        # Top points must have x-coordinates in valid_dx (distance from (0,0) is integer)
        candidates_top = [dx for dx in sorted(valid_dx) if abs(dx) <= search_limit]
        
        if len(candidates_top) < 3:
            continue
        
        # Try combinations of 3 top x-coordinates
        for t1, t2, t3 in combinations(candidates_top, 3):
            if time.time() > start + time_limit:
                break
            
            # For each bottom point (besides (0,0)), it must have integer distance to all 3 top points
            # So for bottom point at x=u: (u-t1)^2 + h^2, (u-t2)^2 + h^2, (u-t3)^2 + h^2 must all be squares
            # This means u-t1, u-t2, u-t3 must be in valid_dx
            
            set_u1 = {t1 + dx for dx in valid_dx}
            set_u2 = {t2 + dx for dx in valid_dx}
            set_u3 = {t3 + dx for dx in valid_dx}
            
            possible_u = set_u1 & set_u2 & set_u3
            possible_u = [u for u in possible_u if abs(u) <= search_limit and u != 0]
            
            if len(possible_u) < 2:
                continue
            
            # Pick 2 more bottom points from possible_u
            for u1, u2 in combinations(possible_u, 2):
                if u1 == u2:
                    continue
                    
                pts = [
                    [[0, 1], [0, 1]],
                    [[u1, 1], [0, 1]],
                    [[u2, 1], [0, 1]],
                    [[t1, 1], [h, 1]],
                    [[t2, 1], [h, 1]],
                    [[t3, 1], [h, 1]]
                ]
                
                # Quick distinct check
                if len(set(tuple(tuple(coord) for coord in p) for p in pts)) < 6:
                    continue
                    
                if _validate(pts, D):
                    return pts
    
    return None

def _try_known_6_point_configurations(D=1):
    """Try some known or likely 6-point configurations."""
    # Try a symmetric configuration
    # Points: (0,0), (a,0), (-a,0), (b,h), (c,h), (d,h)
    # Or other symmetric patterns
    
    # Known: try points from literature
    # A 6-point integral set: (0,0), (15,0), (20,0), (12,9), (16,12), (25,0) - need to verify
    candidates = [
        # Try various small configurations
        [[0,0], [15,0], [20,0], [12,9], [16,12], [25,0]],
        [[0,0], [9,0], [12,0], [6,8], [8,6], [15,0]],
        [[0,0], [7,0], [24,0], [15,8], [18,12], [20,0]],
    ]
    
    for cand in candidates:
        pts = [[[x, 1], [y, 1]] for x, y in cand]
        if len(set(tuple(tuple(coord) for coord in p) for p in pts)) < 6:
            continue
        if _validate(pts, D):
            return pts
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with the known 5-point solution
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    
    deadline = _START_TIME + 140.0
    
    # First try known 6-point configurations (fast)
    if time.time() < deadline - 5:
        pts6_known = _try_known_6_point_configurations(D=1)
        if pts6_known:
            best_score = 6
            best_result = {'D': 1, 'points': pts6_known}
    
    # If not found, do systematic search
    if best_score < 6 and time.time() < deadline - 10:
        remaining_time = deadline - time.time() - 10
        pts6 = _find_6_point_set(D=1, search_limit=300, time_limit=remaining_time)
        if pts6:
            if _validate(pts6, 1):
                best_score = 6
                best_result = {'D': 1, 'points': pts6}
    
    # Final validation
    if best_result is None:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
        
    return best_result

# EVOLVE-BLOCK-END
