"""Combined program: search for 20 integer points with no four collinear and a small no-three-collinear subset."""
import time
import random
from math import gcd

_DEADLINE_SECONDS = 140.0  # safety margin before the 150s kill
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def _collinear(a, b, c):
    # integer cross product test
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) == 0


def _has_four_collinear(points):
    # returns True if any 4 of the points are collinear
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        # For each point, count slopes to others; if any slope has 3+ points, we have 4 collinear
        slope_counts = {}
        for j in range(n):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx == 0 and dy == 0:
                continue
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dxn = dx // g
            dyn = dy // g
            # Canonical sign
            if dxn < 0 or (dxn == 0 and dyn < 0):
                dxn, dyn = -dxn, -dyn
            key = (dxn, dyn)
            slope_counts[key] = slope_counts.get(key, 0) + 1
        for cnt in slope_counts.values():
            if cnt >= 3:
                return True
    return False


def _max_no_three_collinear(points):
    """Exact branch and bound for largest subset with no three collinear."""
    n = len(points)
    # Precompute collinearity for pairs
    collinear = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    collinear[i][j] = True
                    collinear[i][k] = True
                    collinear[j][k] = True

    best = [0]
    chosen = [0] * n

    # Precompute for each point, which other points it is collinear with
    coll_with = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and collinear[i][j]:
                coll_with[i].append(j)

    def can_add(i):
        # Check if adding i creates a bad triple
        # A bad triple (a, b, i) exists if a and b are both chosen and collinear with each other and with i
        # Since collinear is transitive in this context (if a~b and a~i and b~i, they're all on one line),
        # actually collinear[a][b] and collinear[a][i] implies collinear[b][i].
        # So we just need: exists a, b in chosen such that collinear[a][b] and collinear[a][i] and collinear[b][i].
        # This is equivalent to: the number of chosen points collinear with i that are also collinear with each other >= 2?
        # No, simpler: for any chosen point a, if collinear[a][i], check if there is another chosen point b collinear with a and i.
        # Since all points collinear with a and i are on the same line (line ai), we just need to count how many chosen points are on line ai.
        # If count >= 2, then adding i makes 3 collinear.
        
        # Optimization: group chosen points by the line they form with i?
        # Simpler: iterate over chosen points.
        for a in range(n):
            if not chosen[a]:
                continue
            if collinear[a][i]:
                # Count how many other chosen points are on line (a, i)
                cnt = 1  # a is one
                for b in range(n):
                    if b == a or not chosen[b]:
                        continue
                    if collinear[a][b] and collinear[b][i]:
                        cnt += 1
                        if cnt >= 2:
                            return False
        return True

    def branch(start, size):
        if size > best[0]:
            best[0] = size
        if start >= n:
            return
        # Upper bound: size + (n - start)
        if size + (n - start) <= best[0]:
            return
        
        # Heuristic: try to include points that are "less conflicting" first?
        # For now, simple order.
        
        # Try to include start
        if can_add(start):
            chosen[start] = 1
            branch(start + 1, size + 1)
            chosen[start] = 0
        
        # Try to exclude start
        branch(start + 1, size)

    branch(0, 0)
    return best[0]


def _valid(points):
    # distinct, in range, no four collinear
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


def _construct_random(seed):
    rng = random.Random(seed)
    # generate random points and check
    for _ in range(200):
        pts = []
        used = set()
        while len(pts) < 20:
            x = rng.randint(-100, 100)
            y = rng.randint(-100, 100)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        if not _has_four_collinear(pts):
            return pts
    return None


def _construct_perturbed_parabola(seed):
    """
    Start with a parabola-like structure and perturb points to introduce collinear triples
    while avoiding 4-collinear. The goal is to lower the max no-three-collinear subset.
    """
    rng = random.Random(seed)
    # Base points: parabola y = x^2
    base = [[x, x * x] for x in range(-10, 10)]
    
    # Perturb some points
    pts = [p[:] for p in base]
    # Randomly move some points to nearby locations
    for _ in range(rng.randint(5, 15)):
        idx = rng.randint(0, 19)
        # Move to a nearby point
        new_x = pts[idx][0] + rng.randint(-5, 5)
        new_y = pts[idx][1] + rng.randint(-5, 5)
        # Clamp to range
        new_x = max(-100, min(100, new_x))
        new_y = max(-100, min(100, new_y))
        # Ensure distinctness
        if (new_x, new_y) not in [(p[0], p[1]) for p in pts if pts.index(p) != idx]:
            pts[idx] = [new_x, new_y]
    
    # Check validity
    if _valid(pts):
        return pts
    return None


def construct():
    global _START
    _START = time.time()
    
    best_points = None
    best_score = 999
    
    # 1. Deterministic baseline: Parabola
    pts_parab = [[x, x * x] for x in range(-10, 10)]
    if _valid(pts_parab):
        s = _max_no_three_collinear(pts_parab)
        if s < best_score:
            best_score = s
            best_points = pts_parab
    
    # 2. Random search
    # We want to find configurations with many collinear triples but no 4-collinear.
    # Random points usually have few collinear triples, so score is high (near 20).
    # We need a strategy that *encourages* collinearity.
    
    # Strategy: Pick a set of lines, and place points on them.
    # However, ensuring no 4 collinear is tricky.
    
    # Alternative: Local search starting from random or parabola.
    # Let's try more random seeds, but also try the perturbed parabola.
    
    for seed in range(10000):
        if _remaining() < 10.0:
            break
        
        # Mix strategies
        if seed % 2 == 0:
            pts = _construct_random(seed)
        else:
            pts = _construct_perturbed_parabola(seed)
        
        if pts is None:
            continue
        
        # Quick check: if it's not valid, skip
        if not _valid(pts):
            continue
        
        # Compute score
        s = _max_no_three_collinear(pts)
        if s < best_score:
            best_score = s
            best_points = pts
            if best_score <= 12:  # Good enough, stop early
                break
        
        # Safety check
        if _remaining() < 5.0:
            break
    
    if best_points is None:
        best_points = pts_parab
    
    return best_points
