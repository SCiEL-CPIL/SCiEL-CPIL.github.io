import time
from fractions import Fraction
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    # Base set of 5 points
    base_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]

    def is_rational_sqrt(val):
        """Check if val is a perfect square of a rational number."""
        if val < 0:
            return None
        if val == 0:
            return Fraction(0)
        num, den = val.numerator, val.denominator
        sqrt_num = int(math.isqrt(num))
        sqrt_den = int(math.isqrt(den))
        if sqrt_num * sqrt_num == num and sqrt_den * sqrt_den == den:
            return Fraction(sqrt_num, sqrt_den)
        return None

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) == 0

    def check_concyclic(p1, p2, p3, p4):
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]) -
                    m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0]) +
                    m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        def det4(m):
            return (m[0][0]*det3([[m[i][j] for j in [1,2,3]] for i in [1,2,3]]) -
                    m[0][1]*det3([[m[i][j] for j in [0,2,3]] for i in [1,2,3]]) +
                    m[0][2]*det3([[m[i][j] for j in [0,1,3]] for i in [1,2,3]]) -
                    m[0][3]*det3([[m[i][j] for j in [0,1,2]] for i in [1,2,3]]))
        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
        ]
        return det4(m) == 0

    def validate_new_point(new_pt, base_pts):
        """Check if new_pt can be added to base_pts maintaining all constraints."""
        # Check rational distances to all base points
        for p in base_pts:
            if is_rational_sqrt(dist_sq(new_pt, p)) is None:
                return False
        # Check no 3 collinear
        for i in range(len(base_pts)):
            for j in range(i+1, len(base_pts)):
                if check_collinear(new_pt, base_pts[i], base_pts[j]):
                    return False
        # Check no 4 concyclic
        for i in range(len(base_pts)):
            for j in range(i+1, len(base_pts)):
                for k in range(j+1, len(base_pts)):
                    if check_concyclic(new_pt, base_pts[i], base_pts[j], base_pts[k]):
                        return False
        return True

    def try_add_point_x_param(base_pts, x_range_num, x_range_den):
        """Try to add a point by varying x coordinate and solving for y via circle intersection."""
        n = len(base_pts)
        # Try all pairs of anchor points
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline:
                    return None
                p0, p1 = base_pts[i], base_pts[j]
                x0, y0 = p0
                x1, y1 = p1
                dx = x1 - x0
                dy = y1 - y0
                if dx == 0 and dy == 0:
                    continue
                # For each candidate x, solve for y using circle through p0,p1
                # The point (x,y) must satisfy:
                # (x-x0)^2 + (y-y0)^2 = r0^2  where r0^2 = (x1-x0)^2 + (y1-y0)^2... no
                # Actually, we need (x,y) to have rational distance to p0 AND p1.
                # Let d0^2 = (x-x0)^2 + (y-y0)^2 = rational square
                # Let d1^2 = (x-x1)^2 + (y-y1)^2 = rational square
                # Subtract: d0^2 - d1^2 = (x-x0)^2 - (x-x1)^2 + (y-y0)^2 - (y-y1)^2
                # = 2x(x1-x0) + x0^2 - x1^2 + 2y(y1-y0) + y0^2 - y1^2
                # So: d0^2 - d1^2 = 2*dx*x + 2*dy*y + (x0^2 - x1^2 + y0^2 - y1^2)
                # Let K = x0^2 - x1^2 + y0^2 - y1^2
                # d0^2 - d1^2 = 2*dx*x + 2*dy*y + K
                # If dx != 0: y = (d0^2 - d1^2 - 2*dx*x - K) / (2*dy)  (if dy != 0)
                # This is getting complex. Let's use a different approach.
                
                # Alternative: parametrize by trying rational x values and checking if
                # the resulting y (from requiring rational distance to p0) also gives
                # rational distance to p1.
                
                if dy == 0:
                    continue  # Skip horizontal pairs for now, handle separately
                
                # For a given x, y^2 = d0^2 - (x-x0)^2 for some rational d0
                # And y^2 = d1^2 - (x-x1)^2 for some rational d1
                # So d0^2 - (x-x0)^2 = d1^2 - (x-x1)^2
                # d0^2 - d1^2 = (x-x0)^2 - (x-x1)^2 = 2x(x1-x0) + x0^2 - x1^2
                # This is automatically satisfied if we pick d0,d1 consistently.
                # So: pick x, pick d0 rational, then y^2 = d0^2 - (x-x0)^2 must be a rational square.
                # Then check if distance to p1 is also rational.
                
                # Let's iterate over x = a/b with small a,b
                for x_num in range(x_range_num):
                    for x_den in range(1, x_range_den+1):
                        if time.time() > deadline:
                            return None
                        x = Fraction(x_num, x_den)
                        # y^2 = d0^2 - (x-x0)^2 for rational d0
                        # Let t = d0^2 - (x-x0)^2, need t to be rational square
                        # d0^2 = t + (x-x0)^2
                        # Let's try small rational d0 values
                        # Actually, let's just try: for each x, check if there exists y such that
                        # both (x,y) and p0 have rational distance, and (x,y) and p1 have rational distance.
                        # y^2 = r0^2 - (x-x0)^2 where r0 is rational
                        # y^2 = r1^2 - (x-x1)^2 where r1 is rational
                        # So r0^2 - r1^2 = (x-x0)^2 - (x-x1)^2 = 2x*dx + x0^2 - x1^2
                        # Let C = 2*x*dx + x0*x0 - x1*x1
                        # r0^2 - r1^2 = C
                        # (r0-r1)(r0+r1) = C
                        # For rational r0, r1, we need C to factor appropriately.
                        
                        # Simpler: just try a range of rational y values
                        # y = c/d, check if dist to p0 and p1 are rational
                        for y_num in range(-x_range_num, x_range_num+1):
                            for y_den in range(1, x_range_den+1):
                                y = Fraction(y_num, y_den)
                                new_pt = (x, y)
                                # Quick check: distinct
                                if any(new_pt == p for p in base_pts):
                                    continue
                                # Check distance to p0
                                d0_sq = (x - x0)**2 + (y - y0)**2
                                if is_rational_sqrt(d0_sq) is None:
                                    continue
                                # Check distance to p1
                                d1_sq = (x - x1)**2 + (y - y1)**2
                                if is_rational_sqrt(d1_sq) is None:
                                    continue
                                # Passed two anchors, now full validation
                                if validate_new_point(new_pt, base_pts):
                                    return new_pt
        return None

    # Strategy 1: Try to add a 6th point using coordinate search
    best_points = base_points[:]
    
    # Try different ranges, starting small
    for scale in [5, 10, 20, 40, 80]:
        if time.time() > deadline:
            break
        new_pt = try_add_point_x_param(base_points, scale, scale)
        if new_pt:
            best_points = base_points + [new_pt]
            break
    
    # Strategy 2: Try known special points
    if len(best_points) == 5:
        candidates = [
            (Fraction(56), Fraction(33)),
            (Fraction(16), Fraction(63)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(66)),
            (Fraction(8), Fraction(33)),
            (Fraction(-8), Fraction(33)),
            (Fraction(32), Fraction(33)),
            (Fraction(-32), Fraction(33)),
            (Fraction(24), Fraction(33)),
            (Fraction(-24), Fraction(33)),
            (Fraction(40), Fraction(33)),
            (Fraction(-40), Fraction(33)),
            (Fraction(48), Fraction(33)),
            (Fraction(-48), Fraction(33)),
            (Fraction(64), Fraction(33)),
            (Fraction(-64), Fraction(33)),
            (Fraction(80), Fraction(33)),
            (Fraction(-80), Fraction(33)),
            (Fraction(96), Fraction(33)),
            (Fraction(-96), Fraction(33)),
        ]
        for cand in candidates:
            if time.time() > deadline:
                break
            if any(cand == p for p in base_points):
                continue
            if validate_new_point(cand, base_points):
                best_points = base_points + [cand]
                break

    # Final validation and return
    if len(best_points) > 5:
        # Full validation
        n = len(best_points)
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if is_rational_sqrt(dist_sq(best_points[i], best_points[j])) is None:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if check_collinear(best_points[i], best_points[j], best_points[k]):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            for l in range(k+1, n):
                                if check_concyclic(best_points[i], best_points[j], best_points[k], best_points[l]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
        if valid:
            return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}

    # Fallback to 5 points
    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in base_points]}
