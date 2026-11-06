"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import isqrt

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _no_four_concyclic_fast(points, D):
    """Check no four points are concyclic using determinant test.
    Points are (x, y_coeff) where actual y = y_coeff * sqrt(D).
    """
    n = len(points)
    if n < 4:
        return True
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    x1, y1c = points[i]
                    x2, y2c = points[j]
                    x3, y3c = points[k]
                    x4, y4c = points[l]
                    
                    # For D=1, y = y_coeff, so standard circle determinant
                    # det = |x, y, x^2+y^2, 1|
                    # For general D, y_actual = y_coeff * sqrt(D)
                    # det = |x, y_coeff*sqrt(D), x^2 + D*y_coeff^2, 1|
                    
                    # Separate into rational and sqrt(D) parts
                    # Row: [x, y_c*sqrt(D), x^2 + D*y_c^2, 1]
                    # Let's compute the 4x4 det as a + b*sqrt(D)
                    
                    # Using cofactor expansion along column 4 (all 1s):
                    # det = C11 - C12 + C13 - C14 where Cij are 3x3 minors
                    # Each 3x3 minor: |x, y_c*sqrt(D), x^2+D*y_c^2|
                    # = x*(y2c*sqrt(D)*v3 - v2*y3c*sqrt(D)) - y1c*sqrt(D)*(x2*v3 - v2*x3) + v1*(x2*y3c*sqrt(D) - y2c*sqrt(D)*x3)
                    # = sqrt(D) * [x1*(y2c*v3 - v2*y3c) - y1c*(x2*v3 - v2*x3) + v1*(x2*y3c - y2c*x3)]
                    
                    def minor3(i1, i2, i3):
                        a, b = points[i1]
                        c, d = points[i2]
                        e, f = points[i3]
                        va = a*a + D*b*b
                        vc = c*c + D*d*d
                        ve = e*e + D*f*f
                        # det = a*(d*sqrt(D)*ve - vc*f*sqrt(D)) - b*sqrt(D)*(c*ve - vc*e) + va*(c*f*sqrt(D) - d*sqrt(D)*e)
                        # = sqrt(D) * [a*(d*ve - vc*f) - b*(c*ve - vc*e) + va*(c*f - d*e)]
                        bracket = a*(d*ve - vc*f) - b*(c*ve - vc*e) + va*(c*f - d*e)
                        return bracket  # This is the coefficient of sqrt(D)
                    
                    m1 = minor3(i, j, k)
                    m2 = minor3(i, j, l)
                    m3 = minor3(i, k, l)
                    m4 = minor3(j, k, l)
                    
                    # Total det = m1 - m2 + m3 - m4 (all multiplied by sqrt(D))
                    total = m1 - m2 + m3 - m4
                    if total == 0:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Known 5-point integral point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    D = 1
    base_frac = [(Fraction(x), Fraction(y)) for x, y in base_points]
    
    best = base_frac
    best_count = 5
    
    # Try known 6-point configurations
    # Known 6-point integral point set (from literature)
    # Let's try: (-120, 119), (0, 0), (120, 119), (-56, 33), (16, 33), (56, 33)
    # Wait, let me verify distances...
    
    # Try a broader search for 6th point
    # Use smarter search: only check points that have integer distance to all base points
    # Precompute: for each base point, find candidate x,y with integer distance
    
    # First, let's try a wider grid with better pruning
    search_range = 400
    
    found_six = False
    for x in range(-search_range, search_range + 1):
        if time.time() > deadline:
            break
        for y in range(-search_range, search_range + 1):
            if time.time() > deadline:
                break
            
            # Quick check: not collinear with any pair
            collinear = False
            for i in range(5):
                for j in range(i+1, 5):
                    x1, y1 = base_frac[i]
                    x2, y2 = base_frac[j]
                    cp = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                    if cp == 0:
                        collinear = True
                        break
                if collinear:
                    break
            if collinear:
                continue
            
            # Check all pairwise distances are integers
            all_ok = True
            for pt in base_frac:
                dx = x - pt[0]
                dy = y - pt[1]
                d2 = dx * dx + dy * dy
                if not _is_square(d2.numerator * d2.denominator):
                    all_ok = False
                    break
            if not all_ok:
                continue
            
            # Check no four concyclic
            candidate = base_frac + [(Fraction(x), Fraction(y))]
            if not _no_four_concyclic_fast(candidate, D):
                continue
            
            # Found a 6-point set!
            best = candidate
            best_count = 6
            found_six = True
            break
        if found_six:
            break
    
    # If we found 6 points, try to find a 7th
    if found_six:
        found_seven = False
        for x in range(-600, 601):
            if time.time() > deadline:
                break
            for y in range(-600, 601):
                if time.time() > deadline:
                    break
                
                # Quick check: not collinear with any pair
                collinear = False
                for i in range(6):
                    for j in range(i+1, 6):
                        x1, y1 = best[i]
                        x2, y2 = best[j]
                        cp = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                        if cp == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check all pairwise distances are integers
                all_ok = True
                for pt in best:
                    dx = x - pt[0]
                    dy = y - pt[1]
                    d2 = dx * dx + dy * dy
                    if not _is_square(d2.numerator * d2.denominator):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                
                # Check no four concyclic
                candidate = best + [(Fraction(x), Fraction(y))]
                if not _no_four_concyclic_fast(candidate, D):
                    continue
                
                # Found a 7-point set!
                best = candidate
                best_count = 7
                found_seven = True
                break
            if found_seven:
                break
    
    # Format output
    result_points = []
    for x, y in best:
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': D, 'points': result_points}
# EVOLVE-BLOCK-END
