import time
import random
import numpy as np
from collections import defaultdict

_START = time.time()
_DEADLINE = 120.0

def _gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a

def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    a = y1 - y2
    b = x2 - x1
    c = -(a * x1 + b * y1)
    g = _gcd(_gcd(a, b), c)
    if g:
        a //= g; b //= g; c //= g
    if a < 0 or (a == 0 and b < 0):
        a = -a; b = -b; c = -c
    return (a, b, c)

def _count4(pts):
    n = len(pts)
    line_set = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            line_set[k].add(i)
            line_set[k].add(j)
    cnt4 = 0
    has5 = False
    for k, s in line_set.items():
        m = len(s)
        if m == 4:
            cnt4 += 1
        elif m >= 5:
            has5 = True
    return cnt4, has5

def _valid(pts):
    if len(set(map(tuple, pts))) != 16:
        return False
    for (x, y) in pts:
        if abs(x) > 1000 or abs(y) > 1000:
            return False
    _, has5 = _count4(pts)
    return not has5

def _score(pts):
    if not _valid(pts):
        return -1
    c, _ = _count4(pts)
    return c

def _grid44():
    return [[x, y] for x in range(4) for y in range(4)]

def _build_design(rng, base_size=4, defect_prob=0.15):
    """
    Build a 4x4 grid with random defects:
    - Start with 4x4 grid
    - Randomly perturb some points to break 5-collinearities
    - Use structured perturbations that preserve 4-collinearities
    """
    pts = _grid44()
    
    # Apply structured perturbations to break 5-collinearities
    # In a 4x4 grid, we have:
    # - 4 rows (4 collinear each)
    # - 4 columns (4 collinear each)
    # - 2 main diagonals (4 collinear each)
    # Total: 10 lines with exactly 4 points
    # But we might have 5+ collinear in some directions
    
    # Perturb points to break any 5-collinearities
    for _ in range(20):
        if time.time() > _START + _DEADLINE:
            break
        idx = rng.randrange(16)
        # Small perturbation
        dx = rng.choice([-2, -1, 1, 2])
        dy = rng.choice([-2, -1, 1, 2])
        nx = pts[idx][0] + dx
        ny = pts[idx][1] + dy
        if abs(nx) <= 1000 and abs(ny) <= 1000:
            # Check if this creates duplicates
            cand = [list(p) for p in pts]
            cand[idx] = [nx, ny]
            if len(set(map(tuple, cand))) == 16:
                _, has5 = _count4(cand)
                if not has5:
                    pts = cand
                    break
    
    return pts

def _incremental_score(pts, line_cache):
    """
    Compute score using cached line information.
    Returns (score, updated_line_cache)
    """
    n = len(pts)
    line_set = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            line_set[k].add(i)
            line_set[k].add(j)
    cnt4 = 0
    has5 = False
    for k, s in line_set.items():
        m = len(s)
        if m == 4:
            cnt4 += 1
        elif m >= 5:
            has5 = True
            break
    if has5:
        return -1, line_set
    return cnt4, line_set

def _local_search_design(start, rng, max_iters, deadline, T0=3.0, cooling=0.999):
    """
    Local search with design-aware moves:
    - Swap two points
    - Move one point
    - Apply small affine perturbation
    """
    cur = [list(p) for p in start]
    cur_s = _score(cur)
    if cur_s < 0:
        cur = _grid44()
        cur_s = _score(cur)
    
    best = [list(p) for p in cur]
    best_s = cur_s
    T = T0
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.5:
            # Swap two points
            i, j = rng.sample(range(16), 2)
            cand = [list(p) for p in cur]
            cand[i], cand[j] = cand[j], cand[i]
        elif move_type < 0.8:
            # Move one point
            idx = rng.randrange(16)
            ss = rng.choice([3, 5, 8, 15])
            dx = rng.randint(-ss, ss)
            dy = rng.randint(-ss, ss)
            if dx == 0 and dy == 0:
                continue
            cand = [list(p) for p in cur]
            cand[idx] = [cur[idx][0] + dx, cur[idx][1] + dy]
        else:
            # Small affine perturbation on a subset
            subset = rng.sample(range(16), rng.randint(2, 4))
            a = rng.randint(-1, 1)
            b = rng.randint(-1, 1)
            c = rng.randint(-1, 1)
            d = rng.randint(-1, 1)
            tx = rng.randint(-5, 5)
            ty = rng.randint(-5, 5)
            cand = [list(p) for p in cur]
            for idx in subset:
                x, y = cand[idx]
                nx = a * x + b * y + tx
                ny = c * x + d * y + ty
                if abs(nx) <= 1000 and abs(ny) <= 1000:
                    cand[idx] = [nx, ny]
        
        # Check validity
        if len(set(map(tuple, cand))) != 16:
            continue
        if any(abs(x) > 1000 or abs(y) > 1000 for x, y in cand):
            continue
        
        nb_s = _score(cand)
        if nb_s < 0:
            continue
        
        if nb_s > cur_s:
            cur = cand
            cur_s = nb_s
            if cur_s > best_s:
                best = [list(p) for p in cur]
                best_s = cur_s
        else:
            d = nb_s - cur_s
            if d < 0 and rng.random() < np.exp(d / max(T, 1e-6)):
                cur = cand
                cur_s = nb_s
        
        T *= cooling
        if it % 500 == 0 and it > 0:
            T = max(T, 1.5)
    
    return best, best_s

def _try_design_variants(rng, deadline, best_pts, best_score):
    """
    Try multiple design variants:
    - Different base grids
    - Different defect patterns
    - Affine transforms of designs
    """
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Variant 1: 4x4 grid with random perturbations
        if trial % 3 == 0:
            pts = _build_design(rng)
        # Variant 2: 4x4 grid with structured perturbations
        elif trial % 3 == 1:
            pts = _grid44()
            # Perturb corner points
            for idx in [0, 3, 12, 15]:
                if rng.random() < 0.5:
                    dx = rng.choice([-1, 1])
                    dy = rng.choice([-1, 1])
                    pts[idx] = [pts[idx][0] + dx, pts[idx][1] + dy]
        # Variant 3: Affine transform of 4x4 grid
        else:
            a = rng.randint(-2, 2)
            b = rng.randint(-2, 2)
            c = rng.randint(-2, 2)
            d = rng.randint(-2, 2)
            if a * d - b * c == 0:
                continue
            tx = rng.randint(-10, 10)
            ty = rng.randint(-10, 10)
            pts = []
            for x, y in _grid44():
                nx = a * x + b * y + tx
                ny = c * x + d * y + ty
                pts.append([nx, ny])
        
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
            # Quick local search
            best_pts, best_score = _local_search_design(best_pts, rng, 200, deadline)
            if time.time() > deadline:
                break
    
    return best_pts, best_score

def construct():
    deadline = _START + _DEADLINE
    rng = random.Random(98765)
    np.random.seed(98765)
    
    # Start with 4x4 grid
    best = _grid44()
    best_s = _score(best)  # Should be 10
    
    # Phase 1: Design-based construction
    if time.time() < deadline:
        b, s = _local_search_design(_grid44(), rng, 30000, deadline)
        if s > best_s:
            best, best_s = b, s
    
    # Phase 2: Try design variants
    if time.time() < deadline:
        best, best_s = _try_design_variants(rng, deadline, best, best_s)
    
    # Phase 3: Refine best with longer local search
    if time.time() < deadline:
        b, s = _local_search_design(best, rng, 50000, deadline)
        if s > best_s:
            best, best_s = b, s
    
    # Final safety
    if not _valid(best):
        best = _grid44()
    
    return best
