"""Search for 20 integer points with no four collinear and a small no-three-collinear subset."""
import time
import random
from math import gcd
from itertools import combinations

_DEADLINE_SECONDS = 135.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    """Check if any 4 points are collinear. Returns True if found."""
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


def _max_no_three_collinear(points, deadline=None):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    if n <= 2:
        return n

    # Precompute forbidden pairs for each point
    forbidden_pairs = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    forbidden_pairs[i].append((j, k))
                    forbidden_pairs[j].append((i, k))
                    forbidden_pairs[k].append((i, j))

    # Convert to bitmask-based for speed
    # For each point i, we store a list of (a, b) pairs where both being chosen forbids i
    # We'll use a more efficient representation: for each i, a dict from a to bitmask of b's
    # Actually, let's just keep the list but sort and use binary search or just iterate
    # For n=20, the list is small enough

    best = [0]

    def can_add(i, chosen):
        for (a, b) in forbidden_pairs[i]:
            if (chosen >> a) & 1 and (chosen >> b) & 1:
                return False
        return True

    def branch(start, size, chosen):
        if deadline is not None and time.time() > deadline:
            return
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        if size + (n - start) <= best[0]:
            return
        # Try include
        if can_add(start, chosen):
            branch(start + 1, size + 1, chosen | (1 << start))
        # Try exclude
        branch(start + 1, size, chosen)

    branch(0, 0, 0)
    return best[0]


def _construct_parabola(rng):
    """Construct points on a parabola y = x^2 + c*x + d with perturbations."""
    pts = []
    used = set()
    # Use x values from -10 to 9
    for x in range(-10, 10):
        y = x * x + rng.randint(-3, 3)
        # Try to find a valid position
        for attempt in range(30):
            px = x + rng.randint(-1, 1)
            py = y + rng.randint(-2, 2)
            if -100 <= px <= 100 and -100 <= py <= 100 and (px, py) not in used:
                used.add((px, py))
                pts.append([px, py])
                break
        else:
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
    return pts if len(pts) == 20 else None


def _construct_random(rng):
    """Random construction avoiding 4-collinear."""
    pts = []
    used = set()
    attempts = 0
    while len(pts) < 20 and attempts < 5000:
        attempts += 1
        x = rng.randint(-50, 50)
        y = rng.randint(-50, 50)
        if (x, y) in used:
            continue
        # Check no 4 collinear with existing points
        new_pts = pts + [[x, y]]
        if _has_four_collinear(new_pts):
            continue
        used.add((x, y))
        pts.append([x, y])
    return pts if len(pts) == 20 else None


def _local_search(points, rng, deadline=None):
    """Local search: try moving points to reduce the no-three-collinear score."""
    pts = [p[:] for p in points]
    current_score = _max_no_three_collinear(pts, deadline=deadline)
    best_score = current_score
    best_pts = [p[:] for p in pts]
    
    n = len(pts)
    for _ in range(2000):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random point to move
        i = rng.randint(0, n - 1)
        orig_i = pts[i][:]
        
        # Try several new positions
        found_improvement = False
        for attempt in range(20):
            if deadline is not None and time.time() > deadline:
                break
            new_x = rng.randint(-60, 60)
            new_y = rng.randint(-60, 60)
            if (new_x, new_y) == (orig_i[0], orig_i[1]):
                continue
            # Check not duplicate
            if any(p[0] == new_x and p[1] == new_y for j, p in enumerate(pts) if j != i):
                continue
            
            pts[i] = [new_x, new_y]
            
            if not _has_four_collinear(pts):
                s = _max_no_three_collinear(pts, deadline=deadline)
                if s < best_score:
                    best_score = s
                    best_pts = [p[:] for p in pts]
                    found_improvement = True
                    break
                else:
                    pts[i] = orig_i
            else:
                pts[i] = orig_i
        
        if not found_improvement:
            pts[i] = orig_i
    
    return best_pts, best_score


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    rng = random.Random(42)
    
    # Strategy 1: Parabola-based construction (most promising)
    for attempt in range(200):
        if _remaining() < 20.0:
            break
        pts = _construct_parabola(rng)
        if pts is None:
            continue
        if not _has_four_collinear(pts):
            s = _max_no_three_collinear(pts, deadline=_START + _DEADLINE_SECONDS)
            if s < best_score:
                best_score = s
                best_points = [p[:] for p in pts]
                if best_score <= 12:
                    break
    
    # Strategy 2: Random construction
    for attempt in range(50):
        if _remaining() < 20.0:
            break
        pts = _construct_random(rng)
        if pts is None:
            continue
        s = _max_no_three_collinear(pts, deadline=_START + _DEADLINE_SECONDS)
        if s < best_score:
            best_score = s
            best_points = [p[:] for p in pts]
            if best_score <= 12:
                break
    
    # Strategy 3: Local search from best found
    if best_points is not None and _remaining() > 15.0:
        improved_pts, improved_score = _local_search(
            best_points, rng, deadline=_START + _DEADLINE_SECONDS - 5.0
        )
        if improved_score < best_score:
            best_score = improved_score
            best_points = improved_pts
    
    # Fallback: simple parabola
    if best_points is None:
        best_points = [[x, x * x] for x in range(-10, 10)]
    
    return best_points
