import time
from math import isqrt
from fractions import Fraction
import sys

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    DEADLINE = 118.0  # safety margin before 120s kill

    def time_ok():
        return time.time() - t0 < DEADLINE

    def is_square(n):
        """Check if non-negative integer n is a perfect square."""
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_no_three_collinear(pts):
        """pts: list of (x_num, x_den, y_num, y_den). No three collinear."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1 = Fraction(pts[i][0], pts[i][1])
                    y1 = Fraction(pts[i][2], pts[i][3])
                    x2 = Fraction(pts[j][0], pts[j][1])
                    y2 = Fraction(pts[j][2], pts[j][3])
                    x3 = Fraction(pts[k][0], pts[k][1])
                    y3 = Fraction(pts[k][2], pts[k][3])
                    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if det == 0:
                        return False
        return True

    def check_no_four_concyclic(pts, D):
        """pts: list of (x_num, x_den, y_num, y_den). No four concyclic."""
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1 = Fraction(pts[i][0], pts[i][1])
                        y1 = Fraction(pts[i][2], pts[i][3])
                        x2 = Fraction(pts[j][0], pts[j][1])
                        y2 = Fraction(pts[j][2], pts[j][3])
                        x3 = Fraction(pts[k][0], pts[k][1])
                        y3 = Fraction(pts[k][2], pts[k][3])
                        x4 = Fraction(pts[l][0], pts[l][1])
                        y4 = Fraction(pts[l][2], pts[l][3])
                        
                        # Circle: x^2 + D*y^2 + ux + vy + w = 0
                        # For each point: ux + vy + w = -(x^2 + D*y^2)
                        # Solve 3x3 for u,v,w from first 3, check 4th
                        def rhs(px, py):
                            return -(px * px + D * py * py)
                        
                        # Matrix rows: [x, y, 1]
                        a1, b1, c1 = x1, y1, Fraction(1)
                        a2, b2, c2 = x2, y2, Fraction(1)
                        a3, b3, c3 = x3, y3, Fraction(1)
                        
                        r1, r2, r3 = rhs(x1, y1), rhs(x2, y2), rhs(x3, y3)
                        
                        # Cramer's rule for 3x3
                        det_main = (a1*(b2*c3 - b3*c2) - b1*(a2*c3 - a3*c2) + c1*(a2*b3 - a3*b2))
                        if det_main == 0:
                            continue
                        
                        det_u = (r1*(b2*c3 - b3*c2) - b1*(r2*c3 - r3*c2) + c1*(r2*b3 - r3*b2))
                        det_v = (a1*(r2*c3 - r3*c2) - r1*(a2*c3 - a3*c2) + c1*(a2*r3 - a3*r2))
                        det_w = (a1*(b2*r3 - b3*r2) - b1*(a2*r3 - a3*r2) + r1*(a2*b3 - a3*b2))
                        
                        u = det_u / det_main
                        v = det_v / det_main
                        w = det_w / det_main
                        
                        # Check 4th point
                        if u * x4 + v * y4 + w != rhs(x4, y4):
                            continue
                        else:
                            return False
        return True

    def validate(pts, D):
        """Full validation."""
        n = len(pts)
        if n < 3:
            return False
        # Check distinctness
        seen = set()
        for p in pts:
            key = (Fraction(p[0], p[1]), Fraction(p[2], p[3]))
            if key in seen:
                return False
            seen.add(key)
        # Check rational distances
        for i in range(n):
            for j in range(i + 1, n):
                dx = Fraction(pts[i][0], pts[i][1]) - Fraction(pts[j][0], pts[j][1])
                dy = Fraction(pts[i][2], pts[i][3]) - Fraction(pts[j][2], pts[j][3])
                d2 = dx * dx + D * dy * dy
                if d2 < 0:
                    return False
                if not (is_square(d2.numerator) and is_square(d2.denominator)):
                    return False
        if not check_no_three_collinear(pts):
            return False
        if not check_no_four_concyclic(pts, D):
            return False
        return True

    def to_output(pts, D):
        """Convert internal representation to output format."""
        out = []
        for (xn, xd, yn, yd) in pts:
            out.append([[xn, xd], [yn, yd]])
        return {'D': D, 'points': out}

    best_pts = [(0,1,0,1), (3,1,0,1), (0,1,4,1)]
    best_D = 1
    best_count = 3

    # Try D=1 with various starting configurations
    # Known 5-point configuration
    seed_pts_5 = [(0, 1, 0, 1), (-56, 1, 33, 1), (-16, 1, 63, 1), (0, 1, 33, 1), (16, 1, 63, 1)]
    
    if validate(seed_pts_5, 1):
        best_pts = seed_pts_5[:]
        best_D = 1
        best_count = 5

    # For D=1, search for 6th point with wider range
    if time_ok() and best_D == 1 and best_count == 5:
        pts = best_pts[:]
        found = False
        # Search in a larger range
        for x in range(-2000, 2001):
            if not time_ok():
                break
            for y in range(-2000, 2001):
                # Check all distances to existing points
                valid = True
                for p in pts:
                    dx = x - p[0]
                    dy = y - p[2]
                    d2 = dx*dx + dy*dy
                    if not is_square(d2):
                        valid = False
                        break
                if not valid:
                    continue
                # Check not collinear with any pair
                new_pts = pts[:] + [(x, 1, y, 1)]
                if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, 1):
                    best_pts = new_pts
                    best_count = 6
                    found = True
                    break
            if found or not time_ok():
                break
        
        # If found 6, try 7th
        if found and time_ok() and best_count == 6:
            pts = best_pts[:]
            for x in range(-2000, 2001):
                if not time_ok():
                    break
                for y in range(-2000, 2001):
                    valid = True
                    for p in pts:
                        dx = x - p[0]
                        dy = y - p[2]
                        d2 = dx*dx + dy*dy
                        if not is_square(d2):
                            valid = False
                            break
                    if not valid:
                        continue
                    new_pts = pts[:] + [(x, 1, y, 1)]
                    if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, 1):
                        best_pts = new_pts
                        best_count = 7
                        break
                if best_count >= 7 or not time_ok():
                    break

    # Try D=2,3,5,6,7,10,11,13,14,15,17,19,21,22,23,26,29,30,31,33,34,35,37,38,39,41,42,43,46,47,51,53,55,57,58,59,61,62,65,67,69,70,71,73,74,77,78,79,82,83,85,86,87,89,90,93,94,95,97
    if time_ok():
        for D_test in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97]:
            if not time_ok():
                break
            # Start with (0,0) and try to find 2 more points
            base = [(0, 1, 0, 1)]
            found_pair = False
            for x1 in range(-50, 51):
                if not time_ok():
                    break
                for y1 in range(-50, 51):
                    if x1 == 0 and y1 == 0:
                        continue
                    d0 = x1*x1 + D_test*y1*y1
                    if not is_square(d0):
                        continue
                    p1 = (x1, 1, y1, 1)
                    # Find third point
                    for x2 in range(-50, 51):
                        if not time_ok():
                            break
                        for y2 in range(-50, 51):
                            if x2 == 0 and y2 == 0:
                                continue
                            if x2 == x1 and y2 == y1:
                                continue
                            d1 = x2*x2 + D_test*y2*y2
                            if not is_square(d1):
                                continue
                            dx = x2 - x1
                            dy = y2 - y1
                            d2 = dx*dx + D_test*dy*dy
                            if not is_square(d2):
                                continue
                            p2 = (x2, 1, y2, 1)
                            pts3 = base + [p1, p2]
                            if check_no_three_collinear(pts3) and check_no_four_concyclic(pts3, D_test):
                                # Found 3 points! Try to add more
                                cur = pts3[:]
                                for _ in range(10):  # try to add up to 10 more
                                    if not time_ok():
                                        break
                                    added = False
                                    for x3 in range(-200, 201):
                                        if not time_ok():
                                            break
                                        for y3 in range(-200, 201):
                                            valid = True
                                            for p in cur:
                                                dx = x3 - p[0]
                                                dy = y3 - p[2]
                                                d2 = dx*dx + D_test*dy*dy
                                                if not is_square(d2):
                                                    valid = False
                                                    break
                                            if not valid:
                                                continue
                                            new_pts = cur + [(x3, 1, y3, 1)]
                                            if check_no_three_collinear(new_pts) and check_no_four_concyclic(new_pts, D_test):
                                                cur = new_pts
                                                added = True
                                                break
                                        if added or not time_ok():
                                            break
                                    if not added:
                                        break
                                if len(cur) > best_count:
                                    best_pts = cur[:]
                                    best_D = D_test
                                    best_count = len(cur)
                                found_pair = True
                                break
                        if found_pair or not time_ok():
                            break
                    if found_pair or not time_ok():
                        break
                if found_pair or not time_ok():
                    break
            if found_pair:
                continue

    return to_output(best_pts, best_D)

# EVOLVE-BLOCK-END
