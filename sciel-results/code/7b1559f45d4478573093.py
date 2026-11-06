import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 115.0

    def _is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def _check_pairwise_distances(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + D * dy * dy
                if not _is_square(dist_sq):
                    return False
        return True

    def _no_three_collinear(points):
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

    def _four_points_concyclic(p1, p2, p3, p4, D):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        a1 = x2 - x1
        b1 = y2 - y1
        c1 = -(x2 * x2 - x1 * x1) - D * (y2 * y2 - y1 * y1)
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = -(x3 * x3 - x1 * x1) - D * (y3 * y3 - y1 * y1)
        det = a1 * b2 - a2 * b1
        if det == 0:
            return False
        rhs = x1 * x1 + D * y1 * y1 - x4 * x4 - D * y4 * y4
        lhs = (c1 * b2 - c2 * b1) * (x4 - x1) + (a1 * c2 - a2 * c1) * (y4 - y1)
        rhs_scaled = det * rhs
        return lhs == rhs_scaled

    def _no_four_concyclic(points, D):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if _four_points_concyclic(points[i], points[j], points[k], points[l], D):
                            return False
        return True

    def _validate(points, D):
        if len(points) < 3:
            return False
        seen = set()
        for p in points:
            if p in seen:
                return False
            seen.add(p)
        if not _check_pairwise_distances(points, D):
            return False
        if not _no_three_collinear(points):
            return False
        if not _no_four_concyclic(points, D):
            return False
        return True

    def _make_result(points, D):
        return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in points]}

    # Known valid 5-point set with D=1
    pts5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    assert _validate(pts5, 1), "Known 5-point set failed validation"
    best = _make_result(pts5, 1)
    best_n = 5

    # Strategy 1: Try to extend the known 5-point set to 6 points with D=1
    if time.time() < deadline - 20:
        base = list(pts5)
        found6 = False
        # Search in a reasonable range
        for x in range(-200, 201):
            if time.time() > deadline - 15:
                break
            for y in range(-200, 201):
                if time.time() > deadline - 15:
                    break
                new_pt = (x, y)
                if new_pt in base:
                    continue
                candidate = base + [new_pt]
                if _validate(candidate, 1):
                    best = _make_result(candidate, 1)
                    best_n = 6
                    found6 = True
                    break
            if found6:
                break
        if best_n >= 6:
            return best

    # Strategy 2: Search for 6 points with D > 1 using small integer coordinates
    # For D>1, the condition is (x1-x2)^2 + D*(y1-y2)^2 = perfect square
    if time.time() < deadline - 10:
        # Try D=2 first, then others
        for D in [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 93, 94, 95]:
            if time.time() > deadline - 8:
                break
            # Greedy construction: start with (0,0), add points one by one
            points = [(0, 0)]
            # Try to add points systematically
            for x in range(-15, 16):
                if time.time() > deadline - 8:
                    break
                for y in range(-15, 16):
                    if time.time() > deadline - 8:
                        break
                    if x == 0 and y == 0:
                        continue
                    new_pt = (x, y)
                    candidate = points + [new_pt]
                    if len(candidate) >= 3 and _validate(candidate, D):
                        points = candidate
                        if len(points) >= 6:
                            best = _make_result(points, D)
                            best_n = len(points)
                            if best_n >= 6:
                                return best
            # Also try a different starting configuration for this D
            # Start with (0,0), (1,0)
            points2 = [(0, 0), (1, 0)]
            for x in range(-15, 16):
                if time.time() > deadline - 8:
                    break
                for y in range(-15, 16):
                    if time.time() > deadline - 8:
                        break
                    if (x, y) in points2:
                        continue
                    new_pt = (x, y)
                    candidate = points2 + [new_pt]
                    if len(candidate) >= 3 and _validate(candidate, D):
                        points2 = candidate
                        if len(points2) >= 6:
                            best = _make_result(points2, D)
                            best_n = len(points2)
                            if best_n >= 6:
                                return best

    # Strategy 3: Try to find 6 points with D=1 using a broader random search
    if time.time() < deadline - 5:
        import random
        random.seed(12345)
        # Generate random candidates
        for _ in range(2000):
            if time.time() > deadline - 3:
                break
            pts = []
            for _ in range(6):
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                pts.append((x, y))
            # Remove duplicates
            pts = list(set(pts))
            if len(pts) < 6:
                continue
            if _validate(pts, 1):
                best = _make_result(pts, 1)
                best_n = 6
                return best

    # Return the best result found
    return best
