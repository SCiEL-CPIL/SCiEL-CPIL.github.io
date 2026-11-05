import time
import random
from math import gcd
from itertools import combinations

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin

def _count_ordinary(points):
    """Count lines containing exactly 2 of the given points (n=12)."""
    n = len(points)
    if n < 2:
        return 0
    line_counts = {}
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
            if g:
                A //= g
                B //= g
                C //= g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            line_counts[key] = line_counts.get(key, 0) + 1
    return sum(1 for c in line_counts.values() if c == 1)

def _collinear_check(points):
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True

def _valid(points):
    if len(points) != 12:
        return False
    seen = set()
    for x, y in points:
        if abs(x) > 10**9 or abs(y) > 10**9:
            return False
        t = (x, y)
        if t in seen:
            return False
        seen.add(t)
    return not _collinear_check(points)

def _line_key(x1, y1, x2, y2):
    """Canonical line key through two points."""
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if (A or B or C) else 1
    if g:
        A //= g
        B //= g
        C //= g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _points_on_line(x1, y1, x2, y2, max_coord=10**9):
    """Find all integer points on the line through (x1,y1) and (x2,y2) within bounds."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy)) if (dx or dy) else 1
    if g:
        dx //= g
        dy //= g
    # Find the range of t such that x1 + t*dx and y1 + t*dy are within bounds
    # We want points near the existing cluster, so limit t to a reasonable range
    points = []
    # Determine bounds for t
    min_t, max_t = -100, 100
    for t in range(min_t, max_t + 1):
        x = x1 + t * dx
        y = y1 + t * dy
        if abs(x) <= max_coord and abs(y) <= max_coord:
            points.append((x, y))
    return points

def _diverse_pool(pts, radius=8):
    """Generate a diverse candidate pool with emphasis on collinear structures."""
    pool = set()
    pts_set = set(tuple(p) for p in pts)
    
    # 1. Nearby integer points
    for p in pts:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                cand = (p[0] + dx, p[1] + dy)
                if cand not in pts_set:
                    pool.add(cand)
    
    # 2. Points on lines through pairs of existing points
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            for pt in _points_on_line(x1, y1, x2, y2):
                if pt not in pts_set:
                    pool.add(pt)
    
    # 3. Intersection points of lines through different pairs
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A1, B1, C1 = _line_key(x1, y1, x2, y2)
            for k in range(len(pts)):
                for l in range(k + 1, len(pts)):
                    if k == i or k == j or l == i or l == j:
                        continue
                    x3, y3 = pts[k]
                    x4, y4 = pts[l]
                    A2, B2, C2 = _line_key(x3, y3, x4, y4)
                    # Solve A1*x + B1*y = C1 and A2*x + B2*y = C2
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    x = (C1 * B2 - C2 * B1) / det
                    y = (A1 * C2 - A2 * C1) / det
                    if abs(x - round(x)) < 1e-9 and abs(y - round(y)) < 1e-9:
                        xi, yi = int(round(x)), int(round(y))
                        if abs(xi) <= 10**9 and abs(yi) <= 10**9 and (xi, yi) not in pts_set:
                            pool.add((xi, yi))
    
    return list(pool)

def _exhaustive_swap(pts, pool, deadline):
    """Try all possible single-point moves and all pair swaps. Return best found."""
    cur = [list(p) for p in pts]
    cur_score = _count_ordinary(cur)
    best = [list(p) for p in cur]
    best_score = cur_score
    cur_set = set(tuple(p) for p in cur)
    
    # Phase 1: Try moving each point to every candidate in pool
    for idx in range(12):
        if time.time() > deadline:
            break
        old_pt = tuple(cur[idx])
        for cand in pool:
            if cand == old_pt or cand in cur_set:
                continue
            new_pts = [list(p) for p in cur]
            new_pts[idx] = list(cand)
            if not _valid(new_pts):
                continue
            new_score = _count_ordinary(new_pts)
            if new_score < best_score:
                best_score = new_score
                best = [list(p) for p in new_pts]
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
    
    # Phase 2: Try swapping pairs of points with candidates
    for i in range(12):
        if time.time() > deadline:
            break
        for j in range(i + 1, 12):
            if time.time() > deadline:
                break
            old_i = tuple(cur[i])
            old_j = tuple(cur[j])
            for cand_i in pool:
                if cand_i == old_i or cand_i in cur_set:
                    continue
                for cand_j in pool:
                    if cand_j == old_j or cand_j in cur_set or cand_j == cand_i:
                        continue
                    new_pts = [list(p) for p in cur]
                    new_pts[i] = list(cand_i)
                    new_pts[j] = list(cand_j)
                    if not _valid(new_pts):
                        continue
                    new_score = _count_ordinary(new_pts)
                    if new_score < best_score:
                        best_score = new_score
                        best = [list(p) for p in new_pts]
                        cur = new_pts
                        cur_score = new_score
                        cur_set = set(tuple(p) for p in cur)
    
    return best, best_score

def _known_configurations():
    """Return several known/diverse starting configurations."""
    configs = []
    
    # 4x3 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])
    
    # 3x4 grid
    configs.append([[x, y] for x in range(3) for y in range(4)])
    
    # 6x2 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])
    
    # 2x6 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])
    
    # 3x3 grid + 3 extra
    base = [[x, y] for x in range(3) for y in range(3)]
    configs.append(base + [[3, 0], [0, 3], [3, 3]])
    
    # Points on a circle (integer approximations)
    configs.append([[3, 4], [4, 3], [5, 0], [0, 5], [-3, 4], [-4, 3],
                    [-5, 0], [0, -5], [3, -4], [4, -3], [-5, 1], [-1, -5]])
    
    # Random spread
    rng = random.Random(42)
    pts = []
    while len(pts) < 12:
        x = rng.randint(-5, 5)
        y = rng.randint(-5, 5)
        if (x, y) not in pts:
            pts.append([x, y])
    configs.append(pts)
    
    # Structured: two clusters
    configs.append([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5],[5,6],[6,6],
                    [10,10],[11,10],[10,11],[11,11]])
    
    # Near-collinear with perturbations
    configs.append([[0,0],[1,0],[2,0],[3,1],[4,0],[5,1],[6,0],[7,1],
                    [8,0],[9,1],[10,0],[11,1]])
    
    # Hexagonal-like arrangement
    configs.append([[0,0],[1,0],[2,0],[3,0],[0,1],[1,1],[2,1],[3,1],
                    [0,2],[1,2],[2,2],[3,2]])
    
    # Diamond shape
    configs.append([[0,0],[1,1],[2,0],[1,-1],[3,1],[4,0],[3,-1],
                    [2,1],[2,-1],[5,1],[5,-1],[4,1]])
    
    return configs

def construct():
    rng = random.Random(12345)
    
    best_pts = None
    best_score = 999
    
    # Try known configurations
    for cfg in _known_configurations():
        if not _valid(cfg):
            continue
        score = _count_ordinary(cfg)
        if score < best_score:
            best_score = score
            best_pts = [list(p) for p in cfg]
    
    if best_pts is None:
        best_pts = [[x, y] for x in range(4) for y in range(3)]
        best_score = _count_ordinary(best_pts)
    
    # Main search loop: exhaustive swap with diverse pool
    remaining_deadline = _DEADLINE - 5
    
    while time.time() < remaining_deadline:
        pool = _diverse_pool(best_pts, radius=6)
        
        # Limit pool size to keep search tractable
        if len(pool) > 200:
            # Sample a diverse subset
            rng.shuffle(pool)
            pool = pool[:200]
        
        new_pts, new_score = _exhaustive_swap(best_pts, pool, remaining_deadline)
        
        if new_score < best_score:
            best_score = new_score
            best_pts = new_pts
        else:
            # No improvement: try a different pool or break
            # Try with larger radius
            pool2 = _diverse_pool(best_pts, radius=10)
            if len(pool2) > 200:
                rng.shuffle(pool2)
                pool2 = pool2[:200]
            new_pts2, new_score2 = _exhaustive_swap(best_pts, pool2, remaining_deadline)
            if new_score2 < best_score:
                best_score = new_score2
                best_pts = new_pts2
            else:
                # Stagnation: try a random restart from a known config
                if time.time() > remaining_deadline - 20:
                    break
                cfg = rng.choice(_known_configurations())
                if _valid(cfg):
                    cfg_score = _count_ordinary(cfg)
                    if cfg_score < best_score + 3:
                        best_pts = [list(p) for p in cfg]
                        best_score = cfg_score
                # Continue loop to try new pool
    
    # Final polish with small pool
    if time.time() < remaining_deadline - 2:
        pool_final = _diverse_pool(best_pts, radius=3)
        if len(pool_final) > 100:
            rng.shuffle(pool_final)
            pool_final = pool_final[:100]
        final_pts, final_score = _exhaustive_swap(best_pts, pool_final, remaining_deadline)
        if final_score < best_score:
            best_score = final_score
            best_pts = final_pts
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
