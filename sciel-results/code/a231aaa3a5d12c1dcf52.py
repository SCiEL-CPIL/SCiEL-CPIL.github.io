"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 140.0


def _line_key(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dy), abs(dx))
    if g == 0:
        return None
    a = dy // g
    b = -dx // g
    if a < 0 or (a == 0 and b < 0):
        a = -a
        b = -b
    c = a * x1 + b * y1
    return (a, b, c)


def _count_ordinary(points):
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    seen = set()
    ordinary = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            key = _line_key(x1, y1, x2, y2)
            if key is None:
                continue
            if key in seen:
                continue
            seen.add(key)
            a, b, c = key
            cnt = 0
            for (px, py) in pts:
                if a * px + b * py == c:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary


def _all_lines(points):
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    line_map = defaultdict(set)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            key = _line_key(x1, y1, x2, y2)
            if key is None:
                continue
            line_map[key].add(i)
            line_map[key].add(j)
    return {k: sorted(v) for k, v in line_map.items()}


def _count_ordinary_from_lines(lines_dict):
    return sum(1 for v in lines_dict.values() if len(v) == 2)


def _try_move(points, idx, new_pt, lines_dict):
    if new_pt[0] == points[idx][0] and new_pt[1] == points[idx][1]:
        return None
    for i in range(len(points)):
        if i != idx and points[i][0] == new_pt[0] and points[i][1] == new_pt[1]:
            return None
    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
        return None

    new_points = [p[:] for p in points]
    new_points[idx] = [new_pt[0], new_pt[1]]
    new_lines = _all_lines(new_points)
    new_count = _count_ordinary_from_lines(new_lines)
    return (new_points, new_count, new_lines)


def _local_search(points, max_iters=20000):
    best = [p[:] for p in points]
    best_lines = _all_lines(best)
    best_count = _count_ordinary_from_lines(best_lines)
    n = len(best)

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for idx in range(n):
            if time.time() > _DEADLINE - 5:
                break
            x, y = best[idx]
            # Try moves in a larger radius
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    result = _try_move(best, idx, new_pt, best_lines)
                    if result is not None:
                        new_points, new_count, new_lines = result
                        if new_count < best_count:
                            best = new_points
                            best_count = new_count
                            best_lines = new_lines
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_count


def _local_search_wide(points, max_iters=10000):
    best = [p[:] for p in points]
    best_lines = _all_lines(best)
    best_count = _count_ordinary_from_lines(best_lines)
    n = len(best)

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for idx in range(n):
            if time.time() > _DEADLINE - 5:
                break
            x, y = best[idx]
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                        continue
                    result = _try_move(best, idx, new_pt, best_lines)
                    if result is not None:
                        new_points, new_count, new_lines = result
                        if new_count < best_count:
                            best = new_points
                            best_count = new_count
                            best_lines = new_lines
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_count


def _find_integer_points_on_line(points, i, j, max_dist=50):
    """Find integer points on the line through points[i] and points[j]."""
    x1, y1 = points[i]
    x2, y2 = points[j]
    dx = x2 - x1
    dy = y2 - y1
    g = math.gcd(abs(dx), abs(dy))
    if g == 0:
        return []
    dx /= g
    dy /= g

    candidates = []
    for t in range(-max_dist, max_dist + 1):
        if t == 0 or t == 1:
            continue
        px = x1 + t * dx
        py = y1 + t * dy
        if px == int(px) and py == int(py):
            px = int(px)
            py = int(py)
            if abs(px) <= 10**9 and abs(py) <= 10**9:
                # Check not already a point
                if not any(p[0] == px and p[1] == py for p in points):
                    candidates.append((px, py))
    return candidates


def _directed_local_search(points, max_iters=5000):
    """Local search that specifically tries to place points on lines with 2+ other points."""
    best = [p[:] for p in points]
    best_lines = _all_lines(best)
    best_count = _count_ordinary_from_lines(best_lines)
    n = len(best)

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False

        # For each point, try to move it to a position on a line with 2+ other points
        for idx in range(n):
            if time.time() > _DEADLINE - 5:
                break

            # Collect lines through pairs of OTHER points
            x, y = best[idx]
            candidates = set()

            for i in range(n):
                if i == idx:
                    continue
                for j in range(i + 1, n):
                    if j == idx:
                        continue
                    if i == idx or j == idx:
                        continue
                    pts_on_line = best_lines.get(_line_key(best[i][0], best[i][1], best[j][0], best[j][1]), [])
                    if len(pts_on_line) >= 2:
                        # This line has at least 2 points (not including idx)
                        # Try to place idx on this line
                        for pt in _find_integer_points_on_line(best, i, j, max_dist=20):
                            candidates.add(pt)

            # Try each candidate
            for new_pt in list(candidates)[:50]:  # Limit candidates
                result = _try_move(best, idx, new_pt, best_lines)
                if result is not None:
                    new_points, new_count, new_lines = result
                    if new_count < best_count:
                        best = new_points
                        best_count = new_count
                        best_lines = new_lines
                        improved = True
                        break
            if improved:
                break

        if not improved:
            # Fallback: small random moves
            for idx in range(n):
                if time.time() > _DEADLINE - 5:
                    break
                x, y = best[idx]
                for _ in range(5):
                    dx = random.randint(-3, 3)
                    dy = random.randint(-3, 3)
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    result = _try_move(best, idx, new_pt, best_lines)
                    if result is not None:
                        new_points, new_count, new_lines = result
                        if new_count < best_count:
                            best = new_points
                            best_count = new_count
                            best_lines = new_lines
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break

    return best, best_count


def construct():
    # Start with a 3x4 grid (known to have 24 ordinary lines)
    points = [[x, y] for x in range(4) for y in range(3)]
    best = [p[:] for p in points]
    best_count = _count_ordinary(best)

    # Try directed local search first (more targeted)
    if time.time() < _DEADLINE - 10:
        pts1, cnt1 = _directed_local_search(best)
        if cnt1 < best_count:
            best = pts1
            best_count = cnt1

    # Try standard local search
    if time.time() < _DEADLINE - 10:
        pts2, cnt2 = _local_search(best)
        if cnt2 < best_count:
            best = pts2
            best_count = cnt2

    # Try wide local search
    if time.time() < _DEADLINE - 10:
        pts3, cnt3 = _local_search_wide(best)
        if cnt3 < best_count:
            best = pts3
            best_count = cnt3

    # Try a few random restarts with directed search
    for restart in range(3):
        if time.time() > _DEADLINE - 10:
            break
        # Create a random starting point near the grid
        rand_pts = []
        for i in range(12):
            x = random.randint(-3, 6)
            y = random.randint(-3, 6)
            while any(p[0] == x and p[1] == y for p in rand_pts):
                x = random.randint(-3, 6)
                y = random.randint(-3, 6)
            rand_pts.append([x, y])

        pts4, cnt4 = _directed_local_search(rand_pts)
        if cnt4 < best_count:
            best = pts4
            best_count = cnt4

        if time.time() > _DEADLINE - 10:
            break
        pts5, cnt5 = _local_search_wide(best)
        if cnt5 < best_count:
            best = pts5
            best_count = cnt5

    return best
# EVOLVE-BLOCK-END
