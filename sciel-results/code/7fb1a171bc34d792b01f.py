"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _line_key(p1, p2):
    """Canonical line key from two points (dx, dy) reduced to primitive direction."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    # Normalize direction
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    g = _gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # Line is determined by direction and intercept. Use normalized (dx,dy) and offset.
    # For a line with direction (dx,dy), the normal is (-dy, dx).
    # The line through p1: -dy*(x - p1[0]) + dx*(y - p1[1]) = 0
    # => -dy*x + dx*y = -dy*p1[0] + dx*p1[1]
    # Normalize the constant term so that gcd of coefficients is 1 and sign is canonical.
    a = -dy
    b = dx
    c = -a * p1[0] - b * p1[1]
    # Normalize (a, b, c)
    g2 = _gcd(_gcd(abs(a), abs(b)), abs(c))
    if g2 > 0:
        a //= g2
        b //= g2
        c //= g2
    # Canonical sign: first non-zero of (a,b) should be positive
    if a < 0 or (a == 0 and b < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _count_exact4(points):
    """Count lines containing exactly 4 of the given points."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    count = 0
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
    return count

def _has_five_collinear(points):
    """Check if any 5 points are collinear."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, pts in line_map.items():
        if len(pts) >= 5:
            return True
    return False

def _is_valid(points):
    """Check: 16 distinct points, |x|,|y| <= 1000, no five collinear."""
    if len(points) != 16:
        return False
    seen = set()
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
        if p in seen:
            return False
        seen.add(p)
    return not _has_five_collinear(points)

def _make_grid():
    return [[x, y] for x in range(4) for y in range(4)]

def _make_hexagonal():
    # Hexagonal lattice arrangement
    pts = []
    for row in range(4):
        offset = row % 2
        for col in range(4):
            x = col + offset
            y = row * 2
            pts.append([x, y])
    return pts

def _make_affine():
    # Affine transform of grid
    base = _make_grid()
    # Apply a simple affine: (x, y) -> (2x+y, x+2y)
    pts = []
    for p in base:
        x, y = p
        nx = 2 * x + y
        ny = x + 2 * y
        pts.append([nx, ny])
    return pts

def _make_mixed():
    # Mix of grid and offset
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x * 3, y * 3 + (x % 2)])
    return pts

def _local_search(points, deadline, rng):
    """Local search: perturb individual points, swap pairs."""
    best = [p[:] for p in points]
    best_score = _count_exact4(best)
    
    while time.time() < deadline:
        improved = False
        n = len(best)
        # Try perturbing each point
        for i in range(n):
            if time.time() > deadline:
                break
            orig = best[i][:]
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    best[i][0] = orig[0] + dx
                    best[i][1] = orig[1] + dy
                    if not _is_valid(best):
                        continue
                    score = _count_exact4(best)
                    if score > best_score:
                        best_score = score
                        improved = True
        # Restore if we broke out
        best[i] = orig if 'orig' in dir() else best[i]
        
        # Try pairwise swaps
        if time.time() < deadline:
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    best[i], best[j] = best[j][:], best[i][:]
                    if _is_valid(best):
                        score = _count_exact4(best)
                        if score > best_score:
                            best_score = score
                            improved = True
                    else:
                        best[i], best[j] = best[j][:], best[i][:]
        
        if not improved:
            # Try random restarts within local neighborhood
            for _ in range(5):
                if time.time() > deadline:
                    break
                cand = [p[:] for p in best]
                for _ in range(3):
                    idx = rng.randrange(n)
                    cand[idx][0] += rng.randint(-5, 5)
                    cand[idx][1] += rng.randint(-5, 5)
                if _is_valid(cand):
                    score = _count_exact4(cand)
                    if score > best_score:
                        best_score = score
                        best = cand
                        improved = True
            if not improved:
                break
    
    return best, best_score

def construct():
    deadline = time.time() + 130  # 20s safety margin
    rng = random.Random(42)
    
    seeds = [
        _make_grid(),
        _make_hexagonal(),
        _make_affine(),
        _make_mixed(),
    ]
    
    best_points = None
    best_score = -1
    
    for seed in seeds:
        if time.time() > deadline:
            break
        if not _is_valid(seed):
            continue
        s = _count_exact4(seed)
        if s > best_score:
            best_score = s
            best_points = [p[:] for p in seed]
    
    # Local search from each seed
    for seed in seeds:
        if time.time() > deadline:
            break
        if not _is_valid(seed):
            continue
        pts, score = _local_search(seed, deadline, rng)
        if score > best_score:
            best_score = score
            best_points = pts
    
    # Additional random seeds
    while time.time() < deadline - 10:
        pts = []
        for _ in range(16):
            x = rng.randint(-50, 50)
            y = rng.randint(-50, 50)
            if [x, y] not in pts:
                pts.append([x, y])
        if len(pts) == 16 and _is_valid(pts):
            s = _count_exact4(pts)
            if s > best_score:
                best_score = s
                best_points = pts
            # Quick local search
            if time.time() < deadline - 5:
                pts2, s2 = _local_search(pts, min(deadline - 5, time.time() + 20), rng)
                if s2 > best_score:
                    best_score = s2
                    best_points = pts2
    
    if best_points is None:
        best_points = _make_grid()
    
    return [[int(p[0]), int(p[1])] for p in best_points]
# EVOLVE-BLOCK-END
