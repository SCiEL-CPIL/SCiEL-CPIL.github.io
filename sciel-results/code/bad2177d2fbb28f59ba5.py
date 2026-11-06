"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _is_squarefree(n):
    if n == 1:
        return True
    if n % 2 == 0:
        if n % 4 == 0:
            return False
        n //= 2
    i = 3
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 2
    return True

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
    
    # dx = x1 - x2, dy = y1 - y2
    dx_num = x1n * x2d - x2n * x1d
    dx_den = x1d * x2d
    
    dy_num = y1n * y2d - y2n * y1d
    dy_den = y1d * y2d
    
    # dist^2 = dx^2 + D * dy^2
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

def _find_6_point_set(D=1, search_limit=200, time_limit=10.0):
    """Search for a 6-point set on two lines y=0 and y=h."""
    start = time.time()
    # Try various heights h
    for h in range(1, 50):
        if time.time() > time_limit:
            break
        # Find all x in [-search_limit, search_limit] such that there exists an integer distance to (x', h)
        # Actually, we want to pick 3 points on y=0 and 3 on y=h.
        # Let's precompute valid x-differences for cross-line distances.
        # dist^2 = dx^2 + h^2 = k^2 => dx^2 = k^2 - h^2
        # So dx must be such that dx^2 + h^2 is a square.
        
        valid_dx = set()
        for k in range(h, 1000): # k > h
            dx2 = k*k - h*h
            if dx2 < 0:
                break
            dx = int(math.isqrt(dx2))
            if dx*dx == dx2:
                valid_dx.add(dx)
                valid_dx.add(-dx)
        
        if not valid_dx:
            continue
            
        # We need to choose x1, x2, x3 for bottom and y1, y2, y3 for top (y=h)
        # such that |x_i - y_j| is in valid_dx for all i,j.
        # And no three collinear, no four concyclic.
        
        # To simplify, let's fix x1=0.
        # Then y1, y2, y3 must be in valid_dx (distances from 0).
        # Let y1, y2, y3 be chosen from valid_dx.
        # Then x2, x3 must be such that |x2-y_j| in valid_dx for j=1,2,3.
        
        candidates_y = [dx for dx in valid_dx if abs(dx) <= search_limit]
        if len(candidates_y) < 3:
            continue
            
        # Try combinations of 3 top points
        for y1, y2, y3 in combinations(candidates_y, 3):
            if time.time() > time_limit:
                break
            # Top points: (y1, h), (y2, h), (y3, h)
            # Bottom points: (0, 0), (x2, 0), (x3, 0)
            # Check if top points are distinct and not collinear with bottom 0?
            # Collinearity check will be done in validation.
            
            # Find x2 such that dist((x2,0), (y1,h)), dist((x2,0), (y2,h)), dist((x2,0), (y3,h)) are integers.
            # This means (x2-y1)^2 + h^2 is square, etc.
            # So x2-y1, x2-y2, x2-y3 must be in valid_dx.
            # Let u = x2. u-y1 in valid_dx => u in y1+valid_dx.
            # u-y2 in valid_dx => u in y2+valid_dx.
            # u-y3 in valid_dx => u in y3+valid_dx.
            
            set_u1 = {y1 + dx for dx in valid_dx}
            set_u2 = {y2 + dx for dx in valid_dx}
            set_u3 = {y3 + dx for dx in valid_dx}
            
            possible_u = set_u1 & set_u2 & set_u3
            possible_u = [u for u in possible_u if abs(u) <= search_limit]
            
            if len(possible_u) < 2:
                continue
                
            # Pick x2, x3 from possible_u, x2 != x3, x2 != 0, x3 != 0
            for x2, x3 in combinations(possible_u, 2):
                if x2 == 0 or x3 == 0:
                    continue
                if x2 == x3:
                    continue
                    
                pts = [
                    [[0, 1], [0, 1]],
                    [[x2, 1], [0, 1]],
                    [[x3, 1], [0, 1]],
                    [[y1, 1], [h, 1]],
                    [[y2, 1], [h, 1]],
                    [[y3, 1], [h, 1]]
                ]
                
                # Quick check: distinct points
                if len(set(tuple(tuple(coord) for coord in p) for p in pts)) < 6:
                    continue
                    
                if _validate(pts, D):
                    return pts
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_result = None
    best_score = 0
    
    # Start with the known 5-point solution
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    
    deadline = _START_TIME + 140.0
    
    # Try to find a 6-point set using the two-line heuristic
    if time.time() < deadline - 10:
        pts6 = _find_6_point_set(D=1, search_limit=100, time_limit=deadline - time.time() - 10)
        if pts6:
            if _validate(pts6, 1):
                best_score = 6
                best_result = {'D': 1, 'points': pts6}
                print("Found 6 points")
                
    # If we found 6, try to push for 7? Or just return 6.
    # Searching for 7 is much harder. Let's stick to 6 for now if found.
    # If not found, we return 5.
    
    # Final validation
    if best_result is None:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
        
    return best_result

# EVOLVE-BLOCK-END
