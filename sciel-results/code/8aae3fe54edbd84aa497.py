"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """Canonical key for the line through p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return None
    # Reduce by gcd
    g = math.gcd(abs(dx), abs(dy))
    if g != 0:
        dx //= g
        dy //= g
    # Canonical direction: first non-zero component positive
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    # Line: dy*x - dx*y = dy*x1 - dx*y1
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _score(points):
    """Score: number of exact-4 lines, minus penalty if any 5+ collinear."""
    n = len(points)
    line_map = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_map[key].add(i)
    
    count = 0
    has_five = False
    for key, pts in line_map.items():
        if len(pts) == 4:
            count += 1
        elif len(pts) >= 5:
            has_five = True
    
    if has_five:
        return -10000 + count
    return count

def _valid(points):
    """Check distinctness and bounds."""
    if len(points) != 16:
        return False
    if len(set(tuple(p) for p in points)) != 16:
        return False
    for p in points:
        if abs(p[0]) > 1000 or abs(p[1]) > 1000:
            return False
    return True

def _random_affine(base_points, rng):
    """Generate a random affine transform of base_points, checking bounds."""
    for _ in range(200):
        # Random 2x2 matrix with small integers
        a = rng.randint(-3, 3)
        b = rng.randint(-3, 3)
        c = rng.randint(-3, 3)
        d = rng.randint(-3, 3)
        # Check determinant non-zero
        det = a * d - b * c
        if det == 0:
            continue
        
        pts = []
        for x, y in base_points:
            nx = a * x + b * y
            ny = c * x + d * y
            pts.append((nx, ny))
        
        # Center and check if it fits
        tx = rng.randint(-500, 500)
        ty = rng.randint(-500, 500)
        
        all_ok = True
        for nx, ny in pts:
            nx2 = nx + tx
            ny2 = ny + ty
            if abs(nx2) > 1000 or abs(ny2) > 1000:
                all_ok = False
                break
        if not all_ok:
            continue
        
        result = [[nx + tx, ny + ty] for nx, ny in pts]
        if _valid(result):
            return result
    return None

def _local_search(points, time_limit=10.0, rng=None):
    """Local search: move one point at a time to improve score."""
    if rng is None:
        rng = random.Random()
    best = [list(p) for p in points]
    best_score = _score(best)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try moving each point to a nearby random location
        for i in range(16):
            if time.time() - start_time >= time_limit:
                break
            for _ in range(3):
                # Random perturbation
                dx = rng.randint(-5, 5)
                dy = rng.randint(-5, 5)
                if dx == 0 and dy == 0:
                    continue
                new_x = best[i][0] + dx
                new_y = best[i][1] + dy
                if abs(new_x) > 1000 or abs(new_y) > 1000:
                    continue
                # Check distinctness
                trial = [list(p) for p in best]
                trial[i] = [new_x, new_y]
                if len(set(tuple(p) for p in trial)) != 16:
                    continue
                
                s = _score(trial)
                if s > best_score:
                    best = trial
                    best_score = s
                    improved = True
                    break
        if not improved:
            # Try larger moves
            for i in range(16):
                if time.time() - start_time >= time_limit:
                    break
                for _ in range(5):
                    dx = rng.randint(-20, 20)
                    dy = rng.randint(-20, 20)
                    if dx == 0 and dy == 0:
                        continue
                    new_x = best[i][0] + dx
                    new_y = best[i][1] + dy
                    if abs(new_x) > 1000 or abs(new_y) > 1000:
                        continue
                    trial = [list(p) for p in best]
                    trial[i] = [new_x, new_y]
                    if len(set(tuple(p) for p in trial)) != 16:
                        continue
                    
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                        improved = True
                        break
            if not improved:
                # Stuck, try a random restart of one point
                i = rng.randint(0, 15)
                trial = [list(p) for p in best]
                trial[i] = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
                if len(set(tuple(p) for p in trial)) == 16:
                    s = _score(trial)
                    if s > best_score:
                        best = trial
                        best_score = s
                    # Even if not improved, sometimes accept to escape local opt
                    elif rng.random() < 0.1:
                        best = trial
                        best_score = s
                if time.time() - start_time >= time_limit:
                    break
    
    return best, best_score

def _grid_based_search(time_limit=20.0, rng=None):
    """Search over structured grid-like configurations."""
    if rng is None:
        rng = random.Random()
    best_pts = None
    best_score = -1
    start_time = time.time()
    
    # Try various grid sizes and spacings
    for size in range(3, 8):
        if time.time() - start_time >= time_limit:
            break
        for sx in range(1, 50):
            if time.time() - start_time >= time_limit:
                break
            for sy in range(1, 50):
                if time.time() - start_time >= time_limit:
                    break
                # Generate grid points
                pts = []
                for i in range(size):
                    for j in range(size):
                        x = i * sx
                        y = j * sy
                        pts.append([x, y])
                if len(pts) != 16:
                    continue
                
                # Check bounds
                if any(abs(p[0]) > 1000 or abs(p[1]) > 1000 for p in pts):
                    continue
                
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = [list(p) for p in pts]
    
    return best_pts, best_score

def _gen_moves(cur, rng, n=16):
    """Generate candidate moves for local search, including line-completion moves."""
    moves = []
    occ = {tuple(p) for p in cur}
    
    # 3-collinear completion: if 3 points are collinear, try to complete the line
    lines = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(cur[i], cur[j])
            if k is not None:
                lines[k].add(i)
    
    for k, idxs in lines.items():
        if len(idxs) == 3:
            il = list(idxs)
            a, b = cur[il[0]], cur[il[1]]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            g = math.gcd(abs(dx), abs(dy))
            if g:
                dx //= g
                dy //= g
            for t in range(-8, 9):
                cand = [a[0] + t * dx, a[1] + t * dy]
                if abs(cand[0]) <= 1000 and abs(cand[1]) <= 1000 and tuple(cand) not in occ:
                    moves.append((il[0], cand))
    
    # local random
    for _ in range(60):
        i = rng.randrange(n)
        p = cur[i]
        cand = [p[0] + rng.randint(-4, 4), p[1] + rng.randint(-4, 4)]
        if abs(cand[0]) <= 1000 and abs(cand[1]) <= 1000 and tuple(cand) not in occ:
            moves.append((i, cand))
    
    # far kick
    i = rng.randrange(n)
    cand = [rng.randint(-1000, 1000), rng.randint(-1000, 1000)]
    if tuple(cand) not in occ:
        moves.append((i, cand))
    
    return moves

def _polish(cur, curescore, rng, time_left_fn):
    """Steepest single-point moves on line-through-pair candidates."""
    improved = True
    n = 16
    while improved and time_left_fn() > 2.0:
        improved = False
        bestmove = None
        bests = curescore
        occ = {tuple(p) for p in cur}
        cands = set()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    a, b = cur[j], cur[k]
                    dx = b[0] - a[0]
                    dy = b[1] - a[1]
                    g = math.gcd(abs(dx), abs(dy))
                    if g:
                        dx //= g
                        dy //= g
                    for t in range(-3, 4):
                        cand = (a[0] + t * dx, a[1] + t * dy)
                        if abs(cand[0]) <= 1000 and abs(cand[1]) <= 1000 and cand not in occ:
                            cands.add((i, cand))
        rng.shuffle(list(cands))
        for i, cand in list(cands):
            if time_left_fn() < 2.0:
                break
            trial = [list(p) for p in cur]
            trial[i] = list(cand)
            s = _score(trial)
            if s > bests:
                bests = s
                bestmove = (i, cand)
        if bestmove:
            i, cand = bestmove
            cur = [list(p) for p in cur]
            cur[i] = list(cand)
            curescore = bests
            improved = True
    return cur, curescore

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    rng = random.Random(20240517)
    
    def time_left():
        return deadline - time.time()
    
    # Start with the 4x4 grid
    base_grid = [[x, y] for x in range(4) for y in range(4)]
    best_pts = [list(p) for p in base_grid]
    best_score = _score(best_pts)
    
    # Try affine transforms of the base grid
    for _ in range(500):
        if time.time() > deadline:
            break
        transformed = _random_affine(base_grid, rng)
        if transformed is not None:
            s = _score(transformed)
            if s > best_score:
                best_score = s
                best_pts = transformed
    
    # Local search from best
    if time.time() < deadline - 5:
        pts, s = _local_search(best_pts, time_limit=min(15.0, time_left() - 2), rng=rng)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Grid-based search
    if time.time() < deadline - 5:
        pts, s = _grid_based_search(time_limit=min(15.0, time_left() - 2), rng=rng)
        if pts is not None and s > best_score:
            best_score = s
            best_pts = pts
    
    # More local search from the best found
    if time.time() < deadline - 3:
        pts, s = _local_search(best_pts, time_limit=min(10.0, time_left() - 1), rng=rng)
        if s > best_score:
            best_score = s
            best_pts = pts
    
    # Final check
    if not _valid(best_pts):
        # Fallback to original grid
        best_pts = [[x, y] for x in range(4) for y in range(4)]
    
    return best_pts

# EVOLVE-BLOCK-END
