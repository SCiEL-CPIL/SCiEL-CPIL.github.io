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


def _find_integer_points_on_line(a, b, c, x_min, x_max, y_min, y_max, max_points=100):
    """
    Find integer points (x, y) on line a*x + b*y = c within bounds.
    """
    points = []
    if a == 0 and b == 0:
        return points
    
    # Check if there are integer solutions
    # a*x + b*y = c
    # If gcd(a,b) doesn't divide c, no integer solutions
    g = math.gcd(abs(a), abs(b))
    if g == 0 or c % g != 0:
        return points
    
    a_prime = a // g
    b_prime = b // g
    c_prime = c // g
    
    # Find one particular solution using extended Euclidean
    def extended_gcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = extended_gcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)
    
    g_ab, x0, y0 = extended_gcd(a_prime, b_prime)
    # a_prime * x0 + b_prime * y0 = g_ab = 1
    # So a_prime * (x0 * c_prime) + b_prime * (y0 * c_prime) = c_prime
    # Thus a * (x0 * c_prime) + b * (y0 * c_prime) = c
    px = x0 * c_prime
    py = y0 * c_prime
    
    # General solution: x = px + b_prime * t, y = py - a_prime * t
    # Find range of t such that x in [x_min, x_max] and y in [y_min, y_max]
    t_min = float('-inf')
    t_max = float('inf')
    
    if b_prime > 0:
        t_min = max(t_min, math.ceil((x_min - px) / b_prime))
        t_max = min(t_max, math.floor((x_max - px) / b_prime))
    elif b_prime < 0:
        t_min = max(t_min, math.ceil((x_max - px) / b_prime))
        t_max = min(t_max, math.floor((x_min - px) / b_prime))
    
    if -a_prime > 0:
        t_min = max(t_min, math.ceil((y_min - py) / (-a_prime)))
        t_max = min(t_max, math.floor((y_max - py) / (-a_prime)))
    elif -a_prime < 0:
        t_min = max(t_min, math.ceil((y_max - py) / (-a_prime)))
        t_max = min(t_max, math.floor((y_min - py) / (-a_prime)))
    
    t_min = max(0, t_min) if t_min > 0 else t_min
    t_max = min(max_points, t_max) if t_max < max_points else t_max
    
    for t in range(int(t_min), int(t_max) + 1):
        x = px + b_prime * t
        y = py - a_prime * t
        if x_min <= x <= x_max and y_min <= y <= y_max:
            points.append((x, y))
        if len(points) >= max_points:
            break
    
    return points


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


def _try_swap(points, idx1, idx2):
    """
    Try swapping points idx1 and idx2. Return (new_points, delta_ordinary) if valid, else None.
    """
    if idx1 == idx2:
        return None
    
    new_points = [p[:] for p in points]
    new_points[idx1], new_points[idx2] = new_points[idx2], new_points[idx1]
    
    old_count = _count_ordinary(points)
    new_count = _count_ordinary(new_points)
    
    return (new_points, new_count - old_count)


def _smart_candidates(points, idx, radius=15):
    """
    Generate smart candidate positions for point idx.
    Include nearby lattice points and points on existing lines.
    """
    x, y = points[idx]
    candidates = set()
    
    # Nearby lattice points
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                candidates.add((nx, ny))
    
    # Points on lines with 2+ other points
    line_counts = _get_line_counts(points)
    for (a, b, c), (cnt, indices) in line_counts.items():
        if cnt >= 2 and idx not in indices:
            # Find integer points on this line near the current position
            x_min = max(-10**9, x - radius * 10)
            x_max = min(10**9, x + radius * 10)
            y_min = max(-10**9, y - radius * 10)
            y_max = min(10**9, y + radius * 10)
            pts_on_line = _find_integer_points_on_line(a, b, c, x_min, x_max, y_min, y_max, max_points=50)
            for (px, py) in pts_on_line:
                if (px, py) != (x, y):
                    candidates.add((px, py))
    
    return list(candidates)


def _local_search(points, max_iters=2000):
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


def _local_search_smart(points, max_iters=1000):
    """
    Local search with smart candidate generation.
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
            candidates = _smart_candidates(best, idx, radius=10)
            random.shuffle(candidates)
            for new_pt in candidates[:50]:  # Limit candidates per point
                if time.time() > _DEADLINE - 5:
                    break
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
        if not improved:
            break
    
    return best, best_count


def _swap_search(points, max_iters=500):
    """
    Search for beneficial swaps.
    """
    best = [p[:] for p in points]
    best_count = _count_ordinary(best)
    n = len(best)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > _DEADLINE - 5:
                break
            for j in range(i + 1, n):
                if time.time() > _DEADLINE - 5:
                    break
                result = _try_swap(best, i, j)
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
    
    # Try basic local search
    if time.time() < _DEADLINE - 10:
        pts1, cnt1 = _local_search(best)
        if cnt1 < best_count:
            best = pts1
            best_count = cnt1
    
    # Try smart local search
    if time.time() < _DEADLINE - 10:
        pts2, cnt2 = _local_search_smart(best)
        if cnt2 < best_count:
            best = pts2
            best_count = cnt2
    
    # Try swap search
    if time.time() < _DEADLINE - 10:
        pts3, cnt3 = _swap_search(best)
        if cnt3 < best_count:
            best = pts3
            best_count = cnt3
    
    # Try a few random restarts with smart search
    for restart in range(5):
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
        
        if time.time() > _DEADLINE - 10:
            break
        pts4, cnt4 = _local_search_smart(rand_pts)
        if cnt4 < best_count:
            best = pts4
            best_count = cnt4
        
        if time.time() > _DEADLINE - 10:
            break
        pts5, cnt5 = _swap_search(best)
        if cnt5 < best_count:
            best = pts5
            best_count = cnt5
    
    return best
# EVOLVE-BLOCK-END
