"""Integral point sets in general position."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140

    def time_ok():
        return time.time() < DEADLINE

    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def check_collinear(p1, p2, p3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1*x1 + y1*y1
        a2, b2, c2 = x2, y2, x2*x2 + y2*y2
        a3, b3, c3 = x3, y3, x3*x3 + y3*y3
        a4, b4, c4 = x4, y4, x4*x4 + y4*y4
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        return det == 0

    def validate_incremental(points, new_point):
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        return True

    # Start with a known 5-point integral set
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    point_set = set(best_points)

    # Generate Pythagorean offsets up to a given radius
    def gen_pyth_tuples(max_coord):
        tuples = []
        for dx in range(-max_coord, max_coord+1):
            for dy in range(-max_coord, max_coord+1):
                if dx == 0 and dy == 0:
                    continue
                d2 = dx*dx + dy*dy
                if is_square(d2):
                    tuples.append((dx, dy))
        return tuples

    pyth_tuples_300 = gen_pyth_tuples(300)

    # Search loop
    search_range = 200
    attempt = 0
    while time_ok():
        attempt += 1
        current_size = len(best_points)
        if current_size >= 8:
            break

        improved = False

        # Strategy 1: Try Pythagorean offsets from each existing point (most targeted)
        for p in best_points:
            if not time_ok():
                break
            for (dx, dy) in pyth_tuples_300:
                cand = (p[0] + dx, p[1] + dy)
                if cand in point_set:
                    continue
                if validate_incremental(best_points, cand):
                    best_points.append(cand)
                    point_set.add(cand)
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # Strategy 2: Random candidates in expanding box
        for x in range(-search_range, search_range + 1, 3):
            if not time_ok():
                break
            for y in range(-search_range, search_range + 1, 3):
                cand = (x, y)
                if cand in point_set:
                    continue
                if validate_incremental(best_points, cand):
                    best_points.append(cand)
                    point_set.add(cand)
                    improved = True
                    break
            if improved:
                break

        if improved:
            search_range = min(search_range + 40, 800)
            continue

        # Strategy 3: Local search near existing points
        for p in best_points:
            if not time_ok():
                break
            for dx in range(-40, 41, 2):
                for dy in range(-40, 41, 2):
                    if dx == 0 and dy == 0:
                        continue
                    cand = (p[0] + dx, p[1] + dy)
                    if cand in point_set:
                        continue
                    if validate_incremental(best_points, cand):
                        best_points.append(cand)
                        point_set.add(cand)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Strategy 4: Symmetric extensions
        for p in list(best_points):
            if not time_ok():
                break
            for cand in [(p[0], -p[1]), (-p[0], p[1]), (-p[0], -p[1]),
                         (p[1], p[0]), (p[1], -p[0]), (-p[1], p[0]), (-p[1], -p[0])]:
                if cand in point_set:
                    continue
                if validate_incremental(best_points, cand):
                    best_points.append(cand)
                    point_set.add(cand)
                    improved = True
                    break
            if improved:
                break

        if improved:
            continue

        # Strategy 5: Integer circle points
        for p in best_points:
            if not time_ok():
                break
            for r in range(1, 120):
                r2 = r * r
                for dx in range(-r, r+1):
                    dy2 = r2 - dx*dx
                    if dy2 < 0:
                        continue
                    dy = isqrt(dy2)
                    if dy*dy != dy2:
                        continue
                    for dy_sign in ([1, -1] if dy != 0 else [0]):
                        for dx_sign in ([1, -1] if dx != 0 else [0]):
                            cand = (p[0] + dx_sign*dx, p[1] + dy_sign*dy)
                            if cand in point_set:
                                continue
                            if validate_incremental(best_points, cand):
                                best_points.append(cand)
                                point_set.add(cand)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            if not time_ok():
                break
            search_range += 60
            if search_range > 1000:
                break

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
# EVOLVE-BLOCK-END
