import time
from fractions import Fraction
from math import isqrt
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 115.0
    
    def to_frac(n, d=1):
        if d == 1:
            return Fraction(n)
        return Fraction(n, d)
    
    def is_rational_square(fr):
        """Check if a non-negative Fraction is a perfect square of a rational."""
        if fr < 0:
            return False
        if fr == 0:
            return True
        num = fr.numerator
        den = fr.denominator
        # num/den must be a perfect square
        r_num = isqrt(num)
        r_den = isqrt(den)
        return r_num * r_num == num and r_den * r_den == den
    
    def dist_sq(p, q, D):
        """p, q are (Fraction, Fraction). Returns Fraction."""
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + D * dy * dy
    
    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
                    if cross == 0:
                        return False
        return True
    
    def no_four_concyclic(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4
                        # Circle: x² + D·y² + Ax + By + C = 0
                        # From p1: x1² + D·y1² + A·x1 + B·y1 + C = 0
                        # From p2: x2² + D·y2² + A·x2 + B·y2 + C = 0
                        # From p3: x3² + D·y3² + A·x3 + B·y3 + C = 0
                        # Subtract p1 from p2: A(x2-x1) + B(y2-y1) = -(x2²-x1²) - D(y2²-y1²)
                        # Subtract p1 from p3: A(x3-x1) + B(y3-y1) = -(x3²-x1²) - D(y3²-y1²)
                        a1 = x2 - x1
                        b1 = y2 - y1
                        c1 = -(x2*x2 - x1*x1) - D*(y2*y2 - y1*y1)
                        a2 = x3 - x1
                        b2 = y3 - y1
                        c2 = -(x3*x3 - x1*x1) - D*(y3*y3 - y1*y1)
                        det = a1 * b2 - a2 * b1
                        if det == 0:
                            # Degenerate (collinear), skip
                            continue
                        # A = (c1*b2 - c2*b1) / det
                        # B = (a1*c2 - a2*c1) / det
                        # Check p4: A*x4 + B*y4 + C = 0
                        # C = -(x1² + D·y1² + A·x1 + B·y1)
                        # So: A*x4 + B*y4 - x1² - D·y1² - A·x1 - B·y1 = 0
                        # A*(x4-x1) + B*(y4-y1) = x1² + D·y1² - x4² - D·y4²
                        A_num = c1 * b2 - c2 * b1
                        B_num = a1 * c2 - a2 * c1
                        rhs = x1*x1 + D*y1*y1 - x4*x4 - D*y4*y4
                        # A_num/det * (x4-x1) + B_num/det * (y4-y1) = rhs
                        # A_num*(x4-x1) + B_num*(y4-y1) = det * rhs
                        lhs = A_num * (x4 - x1) + B_num * (y4 - y1)
                        if lhs == det * rhs:
                            return False
        return True
    
    def validate(points, D):
        n = len(points)
        if n < 3:
            return False
        # Check distinct
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
        # Check pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq(points[i], points[j], D)
                if not is_rational_square(d2):
                    return False
        # Check no three collinear
        if not no_three_collinear(points):
            return False
        # Check no four concyclic
        if not no_four_concyclic(points, D):
            return False
        return True
    
    def format_points(points, D):
        return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in points]}
    
    # Known 5-point solution for D=1
    pts5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]
    
    if validate(pts5, 1):
        best = format_points(pts5, 1)
    else:
        best = None
    
    # Try to find 6+ points
    # Strategy: For D=2, generate displacement vectors (dx, dy) with dx²+2dy² = rational square
    # Then do a greedy/branch-and-bound search
    
    # Precompute valid displacements for D=2
    # dx² + 2dy² = r², with dx, dy, r rational
    # Parametrize: let dx = a²-2b², dy = 2ab, r = a²+2b² (from (a+b√(-2))... no)
    # Actually x²+2y²=z²: parametrize as x = m²-2n², y = 2mn, z = m²+2n²
    # Check: (m²-2n²)² + 2(2mn)² = m⁴-4m²n²+4n⁴+8m²n² = m⁴+4m²n²+4n⁴ = (m²+2n²)² ✓
    
    D = 2
    displacements = set()
    max_param = 12
    for m in range(1, max_param + 1):
        for n in range(0, max_param + 1):
            if m == 0 and n == 0:
                continue
            dx = Fraction(m * m - 2 * n * n, 1)
            dy = Fraction(2 * m * n, 1)
            # Also negative and swapped variants
            for sx in [1, -1]:
                for sy in [1, -1]:
                    displacements.add((sx * dx, sy * dy))
            # Also the "other" parametrization if exists
            # x²+2y²=z² also has: if we swap roles... no, it's fixed
            # But we can also have dy²*2 + dx² = z², which is the same equation
            # The parametrization covers all primitive solutions
    
    # Add some rational displacements (scale by 1/k)
    rational_displacements = set(displacements)
    for (dx, dy) in list(displacements):
        for k in [2, 3, 4]:
            rational_displacements.add((dx / k, dy / k))
    
    # Greedy search: start with (0,0), add points one by one
    def greedy_search(D, target_n, max_points_to_try, time_limit):
        """Try to build a set of target_n points greedily."""
        start = time.time()
        best_found = None
        
        # Start with origin
        base = [(Fraction(0), Fraction(0))]
        
        # Sort displacements by "size" to try small ones first
        def size(d):
            return abs(d[0]) + abs(d[1])
        
        sorted_disps = sorted(rational_displacements, key=size)
        
        # Try to add points
        for i in range(1, target_n):
            found = False
            for (dx, dy) in sorted_disps:
                if time.time() - start > time_limit:
                    break
                new_pt = (base[0][0] + dx, base[0][1] + dy)
                # Check if new_pt is distinct from all existing
                if new_pt in base:
                    continue
                # Check distance to all existing points
                valid = True
                for existing in base:
                    d2 = dist_sq(new_pt, existing, D)
                    if not is_rational_square(d2):
                        valid = False
                        break
                if not valid:
                    continue
                # Check no three collinear with new point
                candidate = base + [new_pt]
                if not no_three_collinear(candidate):
                    continue
                # Check no four concyclic
                if len(candidate) >= 4 and not no_four_concyclic(candidate, D):
                    continue
                base = candidate
                found = True
                break
            if not found:
                break
            if time.time() - start > time_limit:
                break
        
        if len(base) > 3:
            if best_found is None or len(base) > len(best_found):
                best_found = base
        
        return best_found
    
    # Try greedy search for D=2
    if time.time() - t0 < deadline - 20:
        result = greedy_search(D, 8, 100, deadline - 20 - (time.time() - t0))
        if result is not None and len(result) > 5:
            if validate(result, D):
                best = format_points(result, D)
    
    # If still only 5, try a different approach:
    # Search for 6 points with D=1 using a wider lattice
    if best is None or len(best['points']) <= 5:
        if time.time() - t0 < deadline - 10:
            # Try to extend the known 5-point set with a 6th
            base_pts = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
                        (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
                        (Fraction(16), Fraction(63))]
            
            # Search for a 6th point
            found_6 = False
            for x_num in range(-200, 201):
                for y_num in range(-200, 201):
                    if time.time() - t0 > deadline - 5:
                        break
                    new_pt = (Fraction(x_num), Fraction(y_num))
                    if new_pt in base_pts:
                        continue
                    # Check distances
                    valid = True
                    for p in base_pts:
                        d2 = dist_sq(new_pt, p, 1)
                        if not is_rational_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    candidate = base_pts + [new_pt]
                    if no_three_collinear(candidate) and no_four_concyclic(candidate, 1):
                        if best is None or len(candidate) > len(best['points']):
                            best = format_points(candidate, 1)
                        found_6 = True
                        break
                if found_6 or time.time() - t0 > deadline - 5:
                    break
    
    # Final fallback: return known 5-point set
    if best is None:
        best = format_points(pts5, 1)
    
    return best
