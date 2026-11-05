"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # Start with the known 5-point configuration
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_set = list(base_points)
    best_D = 1
    
    def is_square(n):
        if n < 0:
            return False
        r = int(n ** 0.5)
        return r * r == n or (r + 1) * (r + 1) == n
    
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
                    # Determinant of the 3x3 matrix
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
                        # Four points are concyclic iff the determinant of the 4x4 matrix is 0
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
    
    # Verify the base set is valid
    if validate(base_points, 1):
        best_set = list(base_points)
    
    def try_add_point(current_points, D=1, max_coord=200, max_iters=None):
        """Try to find a new point to add to current_points using systematic search."""
        n = len(current_points)
        if max_iters is None:
            max_iters = (2 * max_coord + 1) ** 2
        iters = 0
        for x in range(-max_coord, max_coord + 1):
            for y in range(-max_coord, max_coord + 1):
                iters += 1
                if iters > max_iters:
                    return None
                if time.time() > deadline - 5:
                    return None
                # Skip if point already exists
                if (x, y) in current_points:
                    continue
                new_points = current_points + [(x, y)]
                if validate(new_points, D):
                    return new_points
        return None
    
    def randomized_search(current_points, D=1, num_trials=10000):
        """Randomized search for a new point."""
        n = len(current_points)
        # Estimate reasonable range based on existing points
        max_x = max(abs(p[0]) for p in current_points) + 100
        max_y = max(abs(p[1]) for p in current_points) + 100
        
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
    
    # Try to add a 6th point using systematic search first
    if time.time() < deadline - 15:
        result = try_add_point(best_set, 1, 150)
        if result is not None:
            best_set = result
    
    # If systematic search failed, try randomized search
    if len(best_set) == 5 and time.time() < deadline - 10:
        result = randomized_search(best_set, 1, 20000)
        if result is not None:
            best_set = result
    
    # If we found 6 points, try to find a 7th
    if len(best_set) >= 6 and time.time() < deadline - 10:
        result = try_add_point(best_set, 1, 100)
        if result is not None:
            best_set = result
    if len(best_set) == 6 and time.time() < deadline - 5:
        result = randomized_search(best_set, 1, 10000)
        if result is not None:
            best_set = result
    
    # If we found 7 points, try to find an 8th
    if len(best_set) >= 7 and time.time() < deadline - 5:
        result = try_add_point(best_set, 1, 80)
        if result is not None:
            best_set = result
    if len(best_set) == 7 and time.time() < deadline - 3:
        result = randomized_search(best_set, 1, 5000)
        if result is not None:
            best_set = result
    
    # Format the result
    points_formatted = []
    for x, y in best_set:
        points_formatted.append([[x, 1], [y, 1]])
    
    return {'D': best_D, 'points': points_formatted}
# EVOLVE-BLOCK-END
