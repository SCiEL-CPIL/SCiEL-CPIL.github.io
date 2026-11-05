"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    def is_square(n):
        if n < 0:
            return False
        if not isinstance(n, int):
            n = int(n)
        r = int(math.isqrt(n)) if n >= 0 else -1
        return r * r == n
    
    def dist_sq(p1, p2, D=1):
        """Return squared distance as a rational number (Fraction)."""
        dx = Fraction(p1[0], 1) - Fraction(p2[0], 1)
        dy = Fraction(p1[1], 1) - Fraction(p2[1], 1)
        return dx * dx + D * dy * dy
    
    def all_dist_rational(points, D=1):
        """Check if all pairwise distances are rational."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                d2 = dist_sq(points[i], points[j], D)
                # d2 must be a perfect square rational
                num = d2.numerator
                den = d2.denominator
                if not is_square(num) or not is_square(den):
                    return False
        return True
    
    def no_three_collinear(points):
        """Check no three points are collinear using exact arithmetic."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if det == 0:
                        return False
        return True
    
    def no_four_concyclic(points):
        """Check no four points are concyclic using exact arithmetic."""
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        a1 = x1 * x1 + y1 * y1
                        a2 = x2 * x2 + y2 * y2
                        a3 = x3 * x3 + y3 * y3
                        a4 = x4 * x4 + y4 * y4
                        det = (
                            a1 * (x2 * (y3 - y4) - y2 * (x3 - x4) + (x3 * y4 - x4 * y3))
                            - x1 * (a2 * (y3 - y4) - y2 * (a3 - a4) + (a3 * y4 - a4 * y3))
                            + y1 * (a2 * (x3 - x4) - x2 * (a3 - a4) + (a3 * x4 - a4 * x3))
                            - 1 * (a2 * (x3 * y4 - x4 * y3) - x2 * (a3 * y4 - a4 * y3) + y2 * (a3 * x4 - a4 * x3))
                        )
                        if det == 0:
                            return False
        return True
    
    def validate(points, D=1):
        """Full validation: all distances rational, no 3 collinear, no 4 concyclic."""
        if not all_dist_rational(points, D):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    # Strategy: Use Eisenstein integer lattice with D=3
    # Points (a, b) in Eisenstein coords map to (x, y) = (a - b/2, b/2) in (x, y*sqrt(3))
    # Norm = a^2 - a*b + b^2 must be perfect square for all pairs
    
    def einstein_norm(a1, b1, a2, b2):
        """Compute squared distance norm in Eisenstein coordinates."""
        da = a1 - a2
        db = b1 - b2
        return da * da - da * db + db * db
    
    def einstein_to_xy(a, b):
        """Convert Eisenstein (a,b) to (x, y) where point is (x, y*sqrt(3))."""
        x = Fraction(a, 1) - Fraction(b, 2)
        y = Fraction(b, 2)
        return (x, y)
    
    # Find valid Eisenstein lattice points where norm from origin is perfect square
    def find_valid_einstein_points(max_ab=50):
        """Find Eisenstein lattice points (a,b) where norm(a,b) is a perfect square."""
        valid = []
        for a in range(-max_ab, max_ab + 1):
            for b in range(-max_ab, max_ab + 1):
                n = einstein_norm(a, b, 0, 0)
                if n >= 0 and is_square(n):
                    valid.append((a, b))
        return valid
    
    # Build a set of points with all pairwise norms perfect squares
    def build_einstein_set(valid_points, max_size=12):
        """Greedy: build a set where all pairwise norms are perfect squares, no 3 collinear, no 4 concyclic."""
        best = [(0, 0)]  # Start with origin
        
        for pt in valid_points:
            if len(best) >= max_size:
                break
            if pt in best:
                continue
            # Check pairwise norm is perfect square
            ok = True
            for existing in best:
                n = einstein_norm(pt[0], pt[1], existing[0], existing[1])
                if n < 0 or not is_square(n):
                    ok = False
                    break
            if not ok:
                continue
            # Convert to xy and check collinearity and concyclicity
            candidate_xy = einstein_to_xy(pt[0], pt[1])
            test_set = [einstein_to_xy(a, b) for a, b in best] + [candidate_xy]
            if no_three_collinear(test_set) and no_four_concyclic(test_set):
                best.append(pt)
        
        return best
    
    # Try D=3 Eisenstein lattice approach
    best_set = []
    best_D = 1
    
    # First, try to find a good set using Eisenstein lattice
    if time.time() < deadline - 20:
        valid_pts = find_valid_einstein_points(60)
        # Sort by norm to prefer smaller points
        valid_pts.sort(key=lambda p: einstein_norm(p[0], p[1], 0, 0))
        einstein_set = build_einstein_set(valid_pts, max_size=12)
        if len(einstein_set) >= 5:
            best_set = [einstein_to_xy(a, b) for a, b in einstein_set]
            best_D = 3
    
    # Fallback: known 5-point configuration with D=1
    if len(best_set) < 5:
        base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        if validate(base_points, 1):
            best_set = list(base_points)
            best_D = 1
    
    # Try to extend the best set found
    def try_add_point_xy(current_points, D, max_coord=200, max_iters=50000):
        """Try to find a new point to add to current_points."""
        iters = 0
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                iters += 1
                if iters > max_iters:
                    return None
                if time.time() > deadline - 5:
                    return None
                if (x, y) in current_points:
                    continue
                new_points = current_points + [(x, y)]
                if validate(new_points, D):
                    return new_points
        return None
    
    def randomized_search_xy(current_points, D, num_trials=10000):
        """Randomized search for a new point."""
        max_x = max(abs(p[0].numerator) for p in current_points) + 100
        max_y = max(abs(p[1].numerator) for p in current_points) + 100
        
        for _ in range(num_trials):
            if time.time() > deadline - 5:
                return None
            x = random.randint(-max_x, max_x)
            y = random.randint(-max_y, max_y)
            if (x, y) in current_points:
                continue
            new_points = current_points + [(x, y)]
            if validate(new_points, D):
                return new_points
        return None
    
    # Try to extend using systematic search
    if len(best_set) < 8 and time.time() < deadline - 15:
        result = try_add_point_xy(best_set, best_D, 100)
        if result is not None:
            best_set = result
    
    if len(best_set) < 8 and time.time() < deadline - 10:
        result = randomized_search_xy(best_set, best_D, 15000)
        if result is not None:
            best_set = result
    
    # Try to extend further
    if len(best_set) < 10 and time.time() < deadline - 10:
        result = try_add_point_xy(best_set, best_D, 80)
        if result is not None:
            best_set = result
    
    if len(best_set) < 10 and time.time() < deadline - 5:
        result = randomized_search_xy(best_set, best_D, 10000)
        if result is not None:
            best_set = result
    
    # Final extension attempts
    if len(best_set) < 12 and time.time() < deadline - 5:
        result = try_add_point_xy(best_set, best_D, 60)
        if result is not None:
            best_set = result
    
    if len(best_set) < 12 and time.time() < deadline - 3:
        result = randomized_search_xy(best_set, best_D, 5000)
        if result is not None:
            best_set = result
    
    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
