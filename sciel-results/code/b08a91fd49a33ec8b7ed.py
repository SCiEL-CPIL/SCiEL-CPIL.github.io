import math
import time
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin from 150s

    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def check_validity(points):
        n = len(points)
        pts_set = set(points)
        if len(pts_set) != n:
            return False
        # All pairwise distances integer
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx * dx + dy * dy
                if not is_square(d2):
                    return False
        # No three collinear
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if area2 == 0:
                        return False
        # No four concyclic
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        x1, y1 = points[i]
                        x2, y2 = points[j]
                        x3, y3 = points[k]
                        x4, y4 = points[l]
                        a1 = x2 - x1
                        b1 = y2 - y1
                        c1 = -(x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1)
                        a2 = x3 - x1
                        b2 = y3 - y1
                        c2 = -(x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1)
                        det = a1 * b2 - a2 * b1
                        if det == 0:
                            continue
                        Df = Fraction(c1 * b2 - c2 * b1, det)
                        Ef = Fraction(a1 * c2 - a2 * c1, det)
                        Ff = -(x1 * x1 + y1 * y1 + Df * x1 + Ef * y1)
                        val = x4 * x4 + y4 * y4 + Df * x4 + Ef * y4 + Ff
                        if val == 0:
                            return False
        return True

    def try_add_point(pts, cand):
        """Check if cand can be added to pts (integer distances, no 3 collinear, no 4 concyclic)."""
        n = len(pts)
        x, y = cand
        # Integer distances to all existing
        for px, py in pts:
            dx = x - px
            dy = y - py
            d2 = dx * dx + dy * dy
            if not is_square(d2):
                return False
        # No 3 collinear with cand
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                cross = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                if cross == 0:
                    return False
        # No 4 concyclic with cand
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a1 = x2 - x1
                    b1 = y2 - y1
                    c1 = -(x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1)
                    a2 = x3 - x1
                    b2 = y3 - y1
                    c2 = -(x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1)
                    det = a1 * b2 - a2 * b1
                    if det == 0:
                        continue
                    Df = Fraction(c1 * b2 - c2 * b1, det)
                    Ef = Fraction(a1 * c2 - a2 * c1, det)
                    Ff = -(x1 * x1 + y1 * y1 + Df * x1 + Ef * y1)
                    val = x * x + y * y + Df * x + Ef * y + Ff
                    if val == 0:
                        return False
        return True

    def find_candidates_two_circles(pts, r_max=200):
        """For each pair of points, find integer points at integer distances from both.
        This is much more targeted than a full grid scan."""
        n = len(pts)
        candidates = set()
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                d12_sq = dx * dx + dy * dy
                if not is_square(d12_sq):
                    continue
                d12 = math.isqrt(d12_sq)
                # Find integer points P = (x,y) such that dist(P, P1) = a (integer) and dist(P, P2) = b (integer)
                # We iterate over possible values of a and b
                # dist(P,P1)^2 = (x-x1)^2 + (y-y1)^2 = a^2
                # dist(P,P2)^2 = (x-x2)^2 + (y-y2)^2 = b^2
                # Subtract: 2*x*(x2-x1) + 2*y*(y2-y1) = a^2 - b^2 + x2^2+y2^2 - x1^2-y1^2
                # This is a linear equation in x,y: A*x + B*y = C
                # where A = 2*(x2-x1), B = 2*(y2-y1), C = a^2 - b^2 + x2^2+y2^2 - x1^2-y1^2
                # For integer x,y on this line, and (x-x1)^2+(y-y1)^2 = a^2
                # We can parametrize the line and check
                A = 2 * dx
                B = 2 * dy
                # gcd of A, B
                g = math.gcd(abs(A), abs(B)) if A != 0 or B != 0 else 1
                if g == 0:
                    continue
                A_reduced = A // g
                B_reduced = B // g
                # C must be divisible by g for integer solutions
                # C = a^2 - b^2 + x2^2+y2^2 - x1^2-y1^2
                base = x2 * x2 + y2 * y2 - x1 * x1 - y1 * y1
                # For each a in some range, b in some range:
                # C = a^2 - b^2 + base
                # C must be divisible by g
                # x = x1 + A_reduced * t + B_reduced * s  ... actually let me use a different approach
                # Parametrize: x = x0 + B_reduced * t, y = y0 - A_reduced * t where (x0,y0) is a particular solution
                # This is getting complex. Let me just do a grid scan but with a smaller range.
                pass
        return candidates

    # Base set of 5 points
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    if not check_validity(base_points):
        fallback = [(0, 0), (3, 0), (0, 4)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback]}

    best_points = list(base_points)

    def search_grid(pts, x_min, x_max, y_min, y_max, time_budget):
        """Grid search for a new point. Returns new set or None."""
        n = len(pts)
        existing = set(pts)
        # Precompute line equations for collinearity check
        lines = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                lines.append((x1, y1, x2 - x1, y2 - y1))
        # Precompute circle equations for concyclicity check
        circles = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    a1 = x2 - x1
                    b1 = y2 - y1
                    c1 = -(x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1)
                    a2 = x3 - x1
                    b2 = y3 - y1
                    c2 = -(x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1)
                    det = a1 * b2 - a2 * b1
                    if det == 0:
                        continue
                    Df = Fraction(c1 * b2 - c2 * b1, det)
                    Ef = Fraction(a1 * c2 - a2 * c1, det)
                    Ff = -(x1 * x1 + y1 * y1 + Df * x1 + Ef * y1)
                    circles.append((Df, Ef, Ff))
        
        for x in range(x_min, x_max + 1):
            if time.time() > time_budget:
                return None
            for y in range(y_min, y_max + 1):
                if time.time() > time_budget:
                    return None
                cand = (x, y)
                if cand in existing:
                    continue
                # Check integer distances
                ok = True
                for px, py in pts:
                    dx = x - px
                    dy = y - py
                    d2 = dx * dx + dy * dy
                    if not is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                # Check collinearity
                collinear = False
                for (x1, y1, dx, dy) in lines:
                    if dx * (y - y1) - dy * (x - x1) == 0:
                        collinear = True
                        break
                if collinear:
                    continue
                # Check concyclicity
                concyclic = False
                for (Df, Ef, Ff) in circles:
                    val = x * x + y * y + Df * x + Ef * y + Ff
                    if val == 0:
                        concyclic = True
                        break
                if concyclic:
                    continue
                return pts + [cand]
        return None

    # Search for 6th point
    if len(best_points) < 6:
        for rng in [100, 200, 400, 800, 1500, 3000]:
            if time.time() > deadline - 15:
                break
            result = search_grid(best_points, -rng, rng, -rng, rng, deadline - 15)
            if result is not None:
                best_points = result
                break

    # Search for 7th point
    if len(best_points) < 7:
        for rng in [100, 200, 400, 800]:
            if time.time() > deadline - 10:
                break
            result = search_grid(best_points, -rng, rng, -rng, rng, deadline - 10)
            if result is not None:
                best_points = result
                break

    # Search for 8th point
    if len(best_points) < 8:
        for rng in [100, 200, 400]:
            if time.time() > deadline - 5:
                break
            result = search_grid(best_points, -rng, rng, -rng, rng, deadline - 5)
            if result is not None:
                best_points = result
                break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
