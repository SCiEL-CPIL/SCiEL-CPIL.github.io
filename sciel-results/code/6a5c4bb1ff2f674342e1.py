import time
import random

def _neighbors(p):
    a, b = p
    return ((a + 1, b), (a - 1, b), (a, b + 1),
            (a, b - 1), (a + 1, b - 1), (a - 1, b + 1))

def _edge_count(points):
    """Count edges in a set of points."""
    pts_set = set(points)
    cnt = 0
    for p in pts_set:
        for q in _neighbors(p):
            if q in pts_set and p < q:
                cnt += 1
    return cnt

def _valid(points):
    """Check validity: 32 distinct points, |a|,|b| <= 64."""
    if len(points) != 32:
        return False
    if len(set(points)) != 32:
        return False
    for p in points:
        if abs(p[0]) > 64 or abs(p[1]) > 64:
            return False
    return True

def _build_hexagonal_spiral(n):
    """Build a hexagonal spiral of n points centered at origin."""
    if n == 1:
        return [(0, 0)]
    points = [(0, 0)]
    radius = 1
    while len(points) < n:
        ring_points = []
        for a in range(-radius, radius + 1):
            for b in range(-radius, radius + 1):
                c = -a - b
                if max(abs(a), abs(b), abs(c)) == radius:
                    ring_points.append((a, b))
        for p in ring_points:
            if len(points) < n:
                points.append(p)
            else:
                break
        radius += 1
    return points

def _local_search(points, pool, deadline, rng):
    """Local search with swap moves and efficient delta computation."""
    best_points = list(points)
    best_score = _edge_count(best_points)
    pts_set = set(best_points)
    iterations = 0
    while time.time() < deadline and iterations < 80000:
        iterations += 1
        p_in = rng.choice(best_points)
        p_out = rng.choice(list(pool))
        if p_in == p_out or p_out in pts_set:
            continue
        edges_lost = 0
        for n in _neighbors(p_in):
            if n in pts_set:
                edges_lost += 1
        edges_gained = 0
        for n in _neighbors(p_out):
            if n in pts_set and n != p_in:
                edges_gained += 1
        delta = edges_gained - edges_lost
        if delta > 0:
            pts_set.remove(p_in)
            pts_set.add(p_out)
            best_points = list(pts_set)
            best_score += delta
        elif delta == 0 and rng.random() < 0.05:
            pts_set.remove(p_in)
            pts_set.add(p_out)
            best_points = list(pts_set)
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    rng = random.Random(42)
    best_points = None
    best_score = -1
    pool = set()
    for a in range(-10, 11):
        for b in range(-10, 11):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 10:
                pool.add((a, b))
    configs = []
    configs.append(_build_hexagonal_spiral(32))
    base3 = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 3:
                base3.append((a, b))
    configs.append(base3[:32])
    base4 = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 4:
                base4.append((a, b))
    rng.shuffle(base4)
    configs.append(base4[:32])
    spiral = _build_hexagonal_spiral(32)
    if spiral:
        idx = rng.randrange(len(spiral))
        a, b = spiral[idx]
        offsets = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
        da, db = offsets[rng.randrange(6)]
        spiral[idx] = (a+da, b+db)
    configs.append(spiral)
    rand_pts = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            c = -a - b
            if max(abs(a), abs(b), abs(c)) <= 5:
                rand_pts.append((a, b))
    rng.shuffle(rand_pts)
    configs.append(rand_pts[:32])
    for initial in configs:
        if time.time() >= deadline - 3.0:
            break
        pts, score = _local_search(initial, pool, deadline - 1.0, rng)
        if score > best_score:
            best_score = score
            best_points = pts
        if best_score >= 76:
            break
    if best_points is None:
        return [[a, b] for a in range(16) for b in range(2)]
    if not _valid(best_points):
        return [[a, b] for a in range(16) for b in range(2)]
    return [[a, b] for a, b in best_points]
