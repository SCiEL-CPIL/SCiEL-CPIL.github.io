import time
import random
from math import gcd
from collections import defaultdict

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before 120 s kill


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
    """Check if all points are collinear."""
    if len(points) < 3:
        return True
    x0, y0 = points[0]
    x1, y1 = points[1]
    for p in points[2:]:
        if (p[0] - x0) * (y1 - y0) != (p[1] - y0) * (x1 - x0):
            return False
    return True


def _valid(points):
    """12 distinct integer points, |coord|<=1e9, not all collinear."""
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


def _make_seed_grid(rows=3, cols=4, ox=0, oy=0, scale=1):
    """rows x cols grid with optional offset and scale."""
    return [[(x * scale + ox), (y * scale + oy)] for x in range(cols) for y in range(rows)]


def _make_seed_triangle():
    """
    Seed: 3 lines of 4 points, intersecting at 3 vertices.
    Line 1: y=0, x in {0, 1, 2, 3} -> (0,0), (1,0), (2,0), (3,0)
    Line 2: x=3, y in {0, 1, 2, 3} -> (3,0), (3,1), (3,2), (3,3)
    Line 3: x+y=3, points (0,3), (1,2), (2,1), (3,0)
    Union:
    (0,0), (1,0), (2,0), (3,0)
    (3,1), (3,2), (3,3)
    (0,3), (1,2), (2,1)
    Total 10 points. Need 2 more.
    Let's add (0,1) and (1,1)?
    Or extend the lines?
    Let's try a different structure: 4 lines of 3 points?
    """
    pts = [
        [0, 0], [1, 0], [2, 0], [3, 0],
        [3, 1], [3, 2], [3, 3],
        [0, 3], [1, 2], [2, 1],
        [0, 1], [1, 1]
    ]
    return pts


def _local_search(points, max_iters=500000):
    """
    Local search with targeted candidate generation.
    """
    cur = [p[:] for p in points]
    cur_score = _count_ordinary(cur)
    rng = random.Random(42)
    iters = 0
    no_improve_count = 0
    
    while iters < max_iters and time.time() < _DEADLINE - 5:
        iters += 1
        
        if no_improve_count > 5000:
            # Perturb a random point to escape local minimum
            idx = rng.randrange(12)
            cur[idx] = [rng.randint(-20, 20), rng.randint(-20, 20)]
            if not _valid(cur):
                cur = [p[:] for p in points] # Reset to valid if needed? Or just retry
                cur_score = _count_ordinary(cur)
            else:
                cur_score = _count_ordinary(cur)
            no_improve_count = 0
            continue

        idx = rng.randrange(12)
        old_pt = cur[idx]
        cur_set = set(tuple(p) for p in cur)
        
        improved = False
        
        # Generate candidates
        # 1. Points on lines defined by other pairs
        candidates = set()
        for j in range(12):
            if j == idx: continue
            for k in range(j+1, 12):
                if k == idx: continue
                p1 = cur[j]
                p2 = cur[k]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                g = gcd(gcd(abs(dx), abs(dy)), 0) if (dx or dy) else 1
                if g:
                    dx //= g
                    dy //= g
                # Extend line by small steps
                for m in range(-3, 4):
                    nx = p1[0] + m * dx
                    ny = p1[1] + m * dy
                    if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                        candidates.add((nx, ny))
        
        # 2. Small perturbations
        for _ in range(5):
            nx = old_pt[0] + rng.randint(-3, 3)
            ny = old_pt[1] + rng.randint(-3, 3)
            candidates.add((nx, ny))

        # 3. Random candidates in bounding box of current points
        min_x = min(p[0] for p in cur)
        max_x = max(p[0] for p in cur)
        min_y = min(p[1] for p in cur)
        max_y = max(p[1] for p in cur)
        for _ in range(5):
            nx = rng.randint(min_x - 2, max_x + 2)
            ny = rng.randint(min_y - 2, max_y + 2)
            candidates.add((nx, ny))

        # Try candidates
        for cand in candidates:
            if cand in cur_set:
                continue
            if abs(cand[0]) > 10**9 or abs(cand[1]) > 10**9:
                continue
            
            new_pts = [p[:] for p in cur]
            new_pts[idx] = list(cand)
            
            if not _valid(new_pts):
                continue
                
            new_score = _count_ordinary(new_pts)
            
            if new_score < cur_score:
                cur = new_pts
                cur_score = new_score
                cur_set = set(tuple(p) for p in cur)
                improved = True
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1

    return cur, cur_score


def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    def _consider(pts):
        nonlocal best_pts, best_score
        if not _valid(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Seed 1: 3x4 Grid
    seed1 = _make_seed_grid(3, 4)
    _consider(seed1)
    
    # Seed 2: Triangle-like structure
    seed2 = _make_seed_triangle()
    _consider(seed2)
    
    # Seed 3: Random points in small box
    rng_init = random.Random(99)
    seed3 = []
    seen = set()
    while len(seed3) < 12:
        x = rng_init.randint(-5, 5)
        y = rng_init.randint(-5, 5)
        if (x, y) not in seen:
            seen.add((x, y))
            seed3.append([x, y])
    _consider(seed3)

    # Local search from best seed
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_ls, score_ls = _local_search(best_pts, max_iters=200000)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # If we still have time, try local search from other seeds if they were close
    if time.time() < _DEADLINE - 10:
        for seed in [seed1, seed2, seed3]:
            if _valid(seed):
                s = _count_ordinary(seed)
                if s <= best_score + 2: # Only try if promising
                    pts_ls, score_ls = _local_search(seed, max_iters=100000)
                    if score_ls < best_score:
                        best_score = score_ls
                        best_pts = pts_ls
                        if best_score == 6:
                            break

    # Final polish
    if best_pts is not None and time.time() < _DEADLINE - 5:
        pts_final, score_final = _local_search(best_pts, max_iters=200000)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = _make_seed_grid(3, 4)

    return best_pts


if __name__ == "__main__":
    pts = construct()
    print("Points:", pts)
    print("Ordinary lines:", _count_ordinary(pts))
    print("Distinct:", len(set(tuple(p) for p in pts)))
    print("Max coord:", max(abs(c) for p in pts for c in p))
    print("Collinear:", _collinear_check(pts))
