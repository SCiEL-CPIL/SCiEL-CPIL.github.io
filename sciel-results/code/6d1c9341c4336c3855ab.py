"""Search for 20 integer points with no four collinear and a small no-three-collinear subset using constructive + local search."""
import time
import random
from math import gcd

_DEADLINE_SECONDS = 140.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[j][0] - points[i][0]
            dy = points[j][1] - points[i][1]
            g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
            if g:
                dxn, dyn = dx // g, dy // g
            else:
                dxn, dyn = dx, dy
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            cnt = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if (points[k][0] - points[i][0]) * dyn == (points[k][1] - points[i][1]) * dxn:
                    cnt += 1
                    if cnt >= 2:
                        return True
    return False


def _max_no_three_collinear(points, timeout=None):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    
    # Precompute collinearity for all pairs
    collinear_with = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    collinear_with[i].add(j)
                    collinear_with[i].add(k)
                    collinear_with[j].add(i)
                    collinear_with[j].add(k)
                    collinear_with[k].add(i)
                    collinear_with[k].add(j)

    best = [0]
    chosen = [0] * n

    def can_add(i):
        # i must not be collinear with any pair already chosen
        for a in range(n):
            if not chosen[a]:
                continue
            for b in collinear_with[a]:
                if b > a and chosen[b]:
                    if i in (a, b) or _collinear(points[a], points[b], points[i]):
                        return False
        return True

    def branch(start, size):
        if timeout is not None and time.time() - timeout > 5.0:
            return
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        if size + (n - start) <= best[0]:
            return
        # Try include
        if can_add(start):
            chosen[start] = 1
            branch(start + 1, size + 1)
            chosen[start] = 0
        # Try exclude
        branch(start + 1, size)

    branch(0, 0)
    return best[0]


def _valid(points):
    if len(points) != 20:
        return False
    seen = set()
    for p in points:
        if not (-100 <= p[0] <= 100 and -100 <= p[1] <= 100):
            return False
        if (p[0], p[1]) in seen:
            return False
        seen.add((p[0], p[1]))
    if _has_four_collinear(points):
        return False
    return True


def _construct_grid_perturbed(rng):
    """Start with a 5x4 grid, perturb to avoid 4-in-line."""
    pts = []
    used = set()
    for row in range(4):
        for col in range(5):
            base_x = -10 + col * 5
            base_y = -7 + row * 5
            # Add small random perturbation
            for _ in range(20):
                x = base_x + rng.randint(-2, 2)
                y = base_y + rng.randint(-2, 2)
                if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in used:
                    used.add((x, y))
                    pts.append([x, y])
                    break
            else:
                # Fallback: just use base
                if (base_x, base_y) not in used:
                    used.add((base_x, base_y))
                    pts.append([base_x, base_y])
    return pts if len(pts) == 20 else None


def _construct_lines(rng):
    """Construct points on lines with exactly 3 per line."""
    # Define several lines, pick 3 points from each, ensure no 4 collinear
    # Use lines with different slopes
    pts = []
    used = set()
    
    # Define candidate lines (as (x1,y1,x2,y2) defining the line)
    lines = []
    for _ in range(15):
        x1 = rng.randint(-50, 50)
        y1 = rng.randint(-50, 50)
        x2 = rng.randint(-50, 50)
        y2 = rng.randint(-50, 50)
        if (x1, y1) != (x2, y2):
            lines.append((x1, y1, x2, y2))
    
    # For each line, find integer points on it
    for (x1, y1, x2, y2) in lines:
        if len(pts) >= 20:
            break
        dx = x2 - x1
        dy = y2 - y1
        g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
        if g == 0:
            continue
        dxn, dyn = dx // g, dy // g
        
        # Find integer points on this line within bounds
        candidates = []
        t_min = max(-100 - x1, -100 - y1) if dxn != 0 else -1000
        t_max = min(100 - x1, 100 - y1) if dxn != 0 else 1000
        if dxn < 0:
            t_min, t_max = t_max, t_min
            t_min, t_max = -t_max, -t_min
        for t in range(max(0, t_min), min(200, t_max) + 1):
            x = x1 + t * dxn
            y = y1 + t * dyn
            if -100 <= x <= 100 and -100 <= y <= 100 and (x, y) not in used:
                candidates.append((x, y))
        
        # Pick up to 3 from candidates
        for (x, y) in candidates[:3]:
            if (x, y) not in used and len(pts) < 20:
                used.add((x, y))
                pts.append([x, y])
    
    # Fill remaining with random points
    while len(pts) < 20:
        x = rng.randint(-100, 100)
        y = rng.randint(-100, 100)
        if (x, y) not in used:
            used.add((x, y))
            pts.append([x, y])
    
    return pts


def _local_search(points, rng, max_iter=500):
    """Local search: try swapping points to reduce the no-three-collinear score."""
    pts = [p[:] for p in points]
    current_score = _max_no_three_collinear(pts, timeout=time.time())
    best_score = current_score
    best_pts = [p[:] for p in pts]
    
    for _ in range(max_iter):
        if _remaining() < 10.0:
            break
        
        # Pick two random points to swap
        i, j = rng.sample(range(20), 2)
        
        # Try moving point i to a new random location
        orig_i = pts[i][:]
        for _ in range(10):
            new_x = rng.randint(-100, 100)
            new_y = rng.randint(-100, 100)
            if (new_x, new_y) in [tuple(p) for p in pts if p != orig_i]:
                continue
            pts[i] = [new_x, new_y]
            
            if not _has_four_collinear(pts):
                s = _max_no_three_collinear(pts, timeout=time.time())
                if s < best_score:
                    best_score = s
                    best_pts = [p[:] for p in pts]
                    current_score = s
                    break
        else:
            pts[i] = orig_i
            continue
        
        if _remaining() < 10.0:
            break
    
    return best_pts, best_score


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    # Try multiple construction strategies
    rng = random.Random(42)
    
    # Strategy 1: Grid perturbed
    for attempt in range(100):
        if _remaining() < 15.0:
            break
        pts = _construct_grid_perturbed(rng)
        if pts is None:
            continue
        if not _has_four_collinear(pts):
            s = _max_no_three_collinear(pts, timeout=time.time())
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                if best_score <= 13:
                    break
    
    # Strategy 2: Lines construction
    for attempt in range(100):
        if _remaining() < 15.0:
            break
        pts = _construct_lines(rng)
        if not _has_four_collinear(pts):
            s = _max_no_three_collinear(pts, timeout=time.time())
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                if best_score <= 13:
                    break
    
    # Strategy 3: Local search from best found
    if best_points is not None and _remaining() > 20.0:
        improved_pts, improved_score = _local_search(best_points, rng, max_iter=200)
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Fallback: parabola
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
