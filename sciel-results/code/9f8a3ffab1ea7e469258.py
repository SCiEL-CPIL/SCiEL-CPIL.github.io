import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s

    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

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
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx * dx + dy * dy
                if not is_square(d2):
                    return False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                    if area2 == 0:
                        return False
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

    if not check_validity(base_points):
        fallback = [(0, 0), (3, 0), (0, 4)]
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in fallback]}

    best_points = list(base_points)

    def try_extend(current_pts, max_range):
        """Search for a new point with integer distances to all current_pts."""
        n = len(current_pts)
        existing = set(current_pts)
        # Precompute for faster distance check
        for x in range(-max_range, max_range + 1):
            if time.time() > deadline:
                return None
            for y in range(-max_range, max_range + 1):
                if time.time() > deadline:
                    return None
                cand = (x, y)
                if cand in existing:
                    continue
                ok = True
                for px, py in current_pts:
                    dx = x - px
                    dy = y - py
                    d2 = dx * dx + dy * dy
                    if not is_square(d2):
                        ok = False
                        break
                if not ok:
                    continue
                # Quick collinearity check
                collinear = False
                for i in range(n):
                    for j in range(i + 1, n):
                        x1, y1 = current_pts[i]
                        x2, y2 = current_pts[j]
                        cross = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
                        if cross == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                new_set = current_pts + [cand]
                if check_validity(new_set):
                    return new_set
        return None

    # Phase 1: Search for 6th point with progressively larger ranges
    for rng in [100, 200, 400, 800, 1500]:
        if time.time() > deadline - 10:
            break
        result = try_extend(best_points, rng)
        if result is not None:
            best_points = result
            break

    # Phase 2: If we have 6, try to find 7th
    if len(best_points) >= 6:
        for rng in [100, 300, 600]:
            if time.time() > deadline - 5:
                break
            result = try_extend(best_points, rng)
            if result is not None:
                best_points = result
                break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
