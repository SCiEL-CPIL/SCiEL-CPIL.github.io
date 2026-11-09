"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before the 150s kill


def _count_ordinary(points):
    """Count lines containing exactly two of the given points."""
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


def _line_through(p1, p2):
    """Return canonical (a, b, c) for line through p1, p2: a*x+b*y=c."""
    x1, y1 = p1
    x2, y2 = p2
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


def _points_on_line(pts, abc):
    """Count points on line a*x+b*y=c."""
    a, b, c = abc
    cnt = 0
    for (px, py) in pts:
        if a * px + b * py == c:
            cnt += 1
    return cnt


def _all_lines(pts):
    """Return dict: line_key -> list of point indices on that line."""
    n = len(pts)
    line_map = {}
    for i in range(n):
        x1, y1 = pts[i]
        for j in range(i + 1, n):
            x2, y2 = pts[j]
            abc = _line_through((x1, y1), (x2, y2))
            if abc is None:
                continue
            if abc not in line_map:
                line_map[abc] = []
            line_map[abc].append(i)
            line_map[abc].append(j)
    # Deduplicate indices in each line
    for abc in line_map:
        line_map[abc] = sorted(set(line_map[abc]))
    return line_map


def _count_ordinary_from_lines(pts):
    """Count ordinary lines given precomputed line map."""
    line_map = _all_lines(pts)
    ordinary = 0
    for abc, indices in line_map.items():
        if len(indices) == 2:
            ordinary += 1
    return ordinary


def _try_move(pts, idx, new_pt, line_map):
    """Try moving point idx to new_pt; return (new_count, new_pts) or None."""
    x, y = new_pt
    # Check if new_pt collides with existing points
    for i, (px, py) in enumerate(pts):
        if i != idx and px == x and py == y:
            return None
    new_pts = list(pts)
    new_pts[idx] = (x, y)
    cnt = _count_ordinary(new_pts)
    return (cnt, new_pts)


def _local_search(pts, max_time=100.0):
    """Local search: try moving each point to various positions."""
    best_pts = list(pts)
    best_count = _count_ordinary(best_pts)
    rng = random.Random(42)
    
    # Candidate positions to try for each point
    # Strategy: try to place points on existing lines (to make them non-ordinary)
    # or form new 3-point lines
    
    deadline = time.time() + max_time
    stuck_count = 0
    
    while time.time() < deadline:
        if stuck_count > 200:
            break
        
        improved = False
        # Try moving each point
        for idx in range(len(best_pts)):
            if time.time() > deadline:
                break
            x0, y0 = best_pts[idx]
            
            # Generate candidate positions
            candidates = []
            
            # Try to align with existing lines: for each pair of other points,
            # extend the line and place point on it
            others = [i for i in range(len(best_pts)) if i != idx]
            for a_i in range(len(others)):
                for b_i in range(a_i + 1, len(others)):
                    p1 = best_pts[others[a_i]]
                    p2 = best_pts[others[b_i]]
                    abc = _line_through(p1, p2)
                    if abc is None:
                        continue
                    a, b, c = abc
                    # Find integer points on this line near (x0, y0)
                    # Line: a*x + b*y = c
                    # Try small offsets
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            nx, ny = x0 + dx, y0 + dy
                            if a * nx + b * ny == c:
                                if (nx, ny) != (x0, y0):
                                    candidates.append((nx, ny))
            
            # Also try random nearby positions
            for _ in range(20):
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                candidates.append((x0 + dx, y0 + dy))
            
            # Try each candidate
            for cand in candidates:
                if time.time() > deadline:
                    break
                result = _try_move(best_pts, idx, cand, None)
                if result is None:
                    continue
                new_count, new_pts = result
                if new_count < best_count:
                    best_count = new_count
                    best_pts = new_pts
                    improved = True
                    stuck_count = 0
                    break
                elif new_count == best_count:
                    # Accept with some probability (sideways move)
                    if rng.random() < 0.1:
                        best_pts = new_pts
                        stuck_count = 0
            
            if improved:
                break
        
        if not improved:
            stuck_count += 1
    
    return best_pts, best_count


def construct():
    # Start with 3 x 4 grid
    points = [(x, y) for x in range(4) for y in range(3)]
    
    # Run local search with time budget
    if time.time() < _DEADLINE:
        points, _ = _local_search(points, max_time=_DEADLINE - time.time() - 5.0)
    
    # Convert to list of lists
    return [[x, y] for x, y in points]
# EVOLVE-BLOCK-END
