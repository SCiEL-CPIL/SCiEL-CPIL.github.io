"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 140.0


def _count_ordinary(points):
    """
    Count lines containing exactly two of the given points.
    """
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    seen = set()
    ordinary = 0
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            if key in seen:
                continue
            seen.add(key)
            cnt = 0
            for (px, py) in pts:
                if a * px + b * py == c:
                    cnt += 1
            if cnt == 2:
                ordinary += 1
    return ordinary


def _get_line_counts(points):
    """
    Return dict: line_key -> (count, list of point indices on that line)
    """
    n = len(points)
    pts = [(p[0], p[1]) for p in points]
    line_map = defaultdict(lambda: (0, []))
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            line_map[key][0] += 1
            line_map[key][1].append(i)
            line_map[key][1].append(j)
    # Fix: each line should be counted once per pair, but we want unique point sets
    # Rebuild properly
    line_map = defaultdict(set)
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            g = math.gcd(abs(dy), abs(dx))
            if g == 0:
                continue
            a = dy // g
            b = -dx // g
            if a < 0 or (a == 0 and b < 0):
                a = -a
                b = -b
            c = a * x1 + b * y1
            key = (a, b, c)
            line_map[key].add(i)
            line_map[key].add(j)
    return {k: (len(v), sorted(v)) for k, v in line_map.items()}


def _try_move(points, idx, new_pt):
    """
    Try moving point idx to new_pt. Return (new_points, delta_ordinary) if valid, else None.
    """
    if new_pt[0] == points[idx][0] and new_pt[1] == points[idx][1]:
        return None
    # Check duplicate
    for i in range(len(points)):
        if i != idx and points[i][0] == new_pt[0] and points[i][1] == new_pt[1]:
            return None
    # Check bounds
    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
        return None
    
    old_pt = points[idx]
    new_points = [p[:] for p in points]
    new_points[idx] = [new_pt[0], new_pt[1]]
    
    old_count = _count_ordinary(points)
    new_count = _count_ordinary(new_points)
    
    return (new_points, new_count - old_count)


def _local_search(points, max_iters=10000):
    """
    Local search: try moving each point to nearby integer positions.
    """
    best = [p[:] for p in points]
    best_count = _count_ordinary(best)
    n = len(best)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for idx in range(n):
            if time.time() > _DEADLINE - 5:
                break
            x, y = best[idx]
            # Try small moves
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    result = _try_move(best, idx, new_pt)
                    if result is not None:
                        new_points, delta = result
                        if delta < 0:
                            best = new_points
                            best_count += delta
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_count


def _generate_candidates_around(points, idx, radius=5):
    """Generate candidate positions near point idx."""
    x, y = points[idx]
    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            candidates.append((x + dx, y + dy))
    return candidates


def _local_search_wide(points, max_iters=5000):
    """
    Wider local search with larger moves.
    """
    best = [p[:] for p in points]
    best_count = _count_ordinary(best)
    n = len(best)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for idx in range(n):
            if time.time() > _DEADLINE - 5:
                break
            x, y = best[idx]
            # Try moves up to radius 10
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx == 0 and dy == 0:
                        continue
                    new_pt = (x + dx, y + dy)
                    if abs(new_pt[0]) > 10**9 or abs(new_pt[1]) > 10**9:
                        continue
                    result = _try_move(best, idx, new_pt)
                    if result is not None:
                        new_points, delta = result
                        if delta < 0:
                            best = new_points
                            best_count += delta
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best, best_count


def construct():
    # Start with 3x4 grid
    points = [[x, y] for x in range(4) for y in range(3)]
    best = [p[:] for p in points]
    best_count = _count_ordinary(best)
    
    # Try local search from this starting point
    if time.time() < _DEADLINE - 10:
        pts1, cnt1 = _local_search(best)
        if cnt1 < best_count:
            best = pts1
            best_count = cnt1
    
    # Try wide local search
    if time.time() < _DEADLINE - 10:
        pts2, cnt2 = _local_search_wide(best)
        if cnt2 < best_count:
            best = pts2
            best_count = cnt2
    
    # Try a few random restarts
    for restart in range(3):
        if time.time() > _DEADLINE - 10:
            break
        # Create a random-ish starting point near the grid
        rand_pts = []
        for i in range(12):
            x = random.randint(-3, 6)
            y = random.randint(-3, 6)
            # Ensure distinct
            while any(p[0] == x and p[1] == y for p in rand_pts):
                x = random.randint(-3, 6)
                y = random.randint(-3, 6)
            rand_pts.append([x, y])
        
        pts3, cnt3 = _local_search(rand_pts)
        if cnt3 < best_count:
            best = pts3
            best_count = cnt3
        
        if time.time() > _DEADLINE - 10:
            break
        pts4, cnt4 = _local_search_wide(best)
        if cnt4 < best_count:
            best = pts4
            best_count = cnt4
    
    return best
# EVOLVE-BLOCK-END
