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
    # Distinctness
    seen = set()
    for p in points:
        key = (tuple(p[0]), tuple(p[1]))
        if key in seen:
            return False
        seen.add(key)
    # Pairwise rational distances
    for i in range(n):
        for j in range(i+1, n):
            if not _check_pairwise_rational(points[i], points[j], D):
                return False
    # No three collinear
    if not _no_three_collinear(points, D):
        return False
    # No four concyclic
    if not _no_four_concyclic(points, D):
        return False
    return True

def _get_valid_dx(h, max_k=2000):
    """Get all valid dx values where dx^2 + h^2 is a perfect square."""
    valid = set()
    for k in range(h + 1, max_k + 1):
        dx2 = k * k - h * h
        if dx2 <= 0:
            continue
        dx = int(math.isqrt(dx2))
        if dx * dx == dx2 and dx > 0:
            valid.add(dx)
            valid.add(-dx)
    return valid

def _find_6_point_set(D=1, time_limit=40.0):
    """
    Search for a 6-point integral point set on two parallel lines.
    """
    start = time.time()
    
    # Try different heights h
    for h in range(1, 50):
        if time.time() - start > time_limit:
            break
        
        valid_dx = _get_valid_dx(h, max_k=3000)
        
        if len(valid_dx) < 10:
            continue
        
        # Filter to reasonable range
        valid_dx_sorted = sorted([dx for dx in valid_dx if abs(dx) <= 500])
        
        if len(valid_dx_sorted) < 6:
            continue
        
        # Strategy: 3 points on y=0 at x = a, b, c
        # 3 points on y=h at x = d, e, f
        # Constraints:
        # - |a-b|, |a-c|, |b-c| must be in valid_dx ∪ {0} (same line, distance = |dx|)
        # - |d-e|, |d-f|, |e-f| must be in valid_dx ∪ {0}
        # - For each bottom x and top x': (x - x')^2 + h^2 must be a perfect square
        #   i.e., (x - x') in valid_dx
        
        # Fix bottom points: try small sets of 3 integers where pairwise diffs are in valid_dx
        # Since valid_dx includes ±dx values, we need a set S of 3 integers where all pairwise diffs are in valid_dx
        
        # Build adjacency: x ~ y if (x-y) in valid_dx
        # Find triangles in this graph among small integers
        
        candidates = list(range(-100, 101))
        
        # Build valid difference set for quick lookup
        valid_set = set(valid_dx)
        
        # Find all pairs (i,j) with i<j and (j-i) in valid_set
        valid_pairs = []
        for ii in range(len(candidates)):
            for jj in range(ii + 1, len(candidates)):
                diff = candidates[jj] - candidates[ii]
                if diff in valid_set:
                    valid_pairs.append((ii, jj))
        
        # Find triangles: three indices i<j<k where all three pairs are valid
        triangles = []
        pair_set = set()
        for ii, jj in valid_pairs:
            pair_set.add((ii, jj))
        
        for ii, jj in valid_pairs:
            for kk in range(jj + 1, len(candidates)):
                if (jj, kk) in pair_set and (ii, kk) in pair_set:
                    triangles.append((ii, jj, kk))
        
        if len(triangles) < 5:
            continue
        
        # For each triangle (bottom), find compatible top points
        # Top point at x_top must satisfy: (x_top - x_bottom) in valid_set for all 3 bottom points
        for tri in triangles[:200]:  # Limit number of triangles to try
            if time.time() - start > time_limit:
                break
            
            bx1, bx2, bx3 = candidates[tri[0]], candidates[tri[1]], candidates[tri[2]]
            
            # Find top x values compatible with all three bottom points
            top_candidates = []
            for tx in range(-200, 201):
                if (tx - bx1) in valid_set and (tx - bx2) in valid_set and (tx - bx3) in valid_set:
                    top_candidates.append(tx)
            
            if len(top_candidates) < 3:
                continue
            
            # Find 3 top points where pairwise differences are in valid_set
            top_triangles = []
            for ti in range(len(top_candidates)):
                for tj in range(ti + 1, len(top_candidates)):
                    if (top_candidates[tj] - top_candidates[ti]) in valid_set:
                        for tk in range(tj + 1, len(top_candidates)):
                            if (top_candidates[tk] - top_candidates[ti]) in valid_set and \
                               (top_candidates[tk] - top_candidates[tj]) in valid_set:
                                top_triangles.append((ti, tj, tk))
            
            if len(top_triangles) < 1:
                continue
            
            # Try first few top triangles
            for tt in top_triangles[:20]:
                if time.time() - start > time_limit:
                    break
                
                tx1, tx2, tx3 = top_candidates[tt[0]], top_candidates[tt[1]], top_candidates[tt[2]]
                
                pts = [
                    [[bx1, 1], [0, 1]],
                    [[bx2, 1], [0, 1]],
                    [[bx3, 1], [0, 1]],
                    [[tx1, 1], [h, 1]],
                    [[tx2, 1], [h, 1]],
                    [[tx3, 1], [h, 1]]
                ]
                
                # Quick distinctness
                if len(set(tuple(tuple(c) for c in p) for p in pts)) < 6:
                    continue
                
                if _validate(pts, D):
                    return pts
    
    return None

def _find_7_point_set(D=1, time_limit=30.0):
    """
    Try to extend a 6-point set to 7 points.
    """
    start = time.time()
    
    # First find a 6-point set
    pts6 = _find_6_point_set(D, time_limit=15.0)
    if not pts6:
        return None
    
    # Determine the structure of pts6
    # Points should be on two lines y=0 and y=h
    y_vals = set()
    for p in pts6:
        y_vals.add((p[1][0], p[1][1]))
    
    if len(y_vals) != 2:
        return None
    
    # Get the two y-values
    y0 = min(y_vals, key=lambda t: t[0]/t[1] if t[1] != 0 else 0)
    y1 = max(y_vals, key=lambda t: t[0]/t[1] if t[1] != 0 else 0)
    
    h_num = y1[0] * y0[1] - y0[0] * y1[1]
    h_den = y1[1] * y0[1]
    
    if h_den == 0 or h_num == 0:
        return None
    
    # Try adding a 7th point on either line
    for target_y in [y0, y1]:
        if time.time() - start > time_limit:
            break
        
        for x in range(-300, 301):
            if time.time() - start > time_limit:
                break
            
            new_pt = [[x, 1], list(target_y)]
            
            # Check if this point is already in the set
            is_dup = False
            for p in pts6:
                if p[0][0] == x and p[0][1] == 1 and p[1][0] == target_y[0] and p[1][1] == target_y[1]:
                    is_dup = True
                    break
            if is_dup:
                continue
            
            candidate = pts6 + [new_pt]
            
            # Quick pairwise check before full validation
            ok = True
            for i in range(6):
                if not _check_pairwise_rational(pts6[i], new_pt, D):
                    ok = False
                    break
            if not ok:
                continue
            
            if _validate(candidate, D):
                return candidate
    
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with the known 5-point solution
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    
    deadline = _START_TIME + 140.0
    
    # Try to find a 6-point set
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 15
        pts6 = _find_6_point_set(D=1, time_limit=remaining)
        if pts6:
            if _validate(pts6, 1):
                best_score = 6
                best_result = {'D': 1, 'points': pts6}
        
        # If we found 6, try to push for 7
        if best_score == 6 and time.time() < deadline - 5:
            remaining = deadline - time.time() - 5
            pts7 = _find_7_point_set(D=1, time_limit=remaining)
            if pts7:
                if _validate(pts7, 1):
                    best_score = 7
                    best_result = {'D': 1, 'points': pts7}
    
    # Final validation
    if best_result is None:
        best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
        
    return best_result

# EVOLVE-BLOCK-END
