"""Combined approach: structured seeds (grids, staggered, diagonal) + fast
single-point local search with incremental scoring and validity enforcement.
Maximizes lines with exactly four points, no five collinear."""
import time
import math
import random
import itertools

_START = time.time()
_DEADLINE = _START + 115.0  # safety margin before the 150 s kill
_BOUND = 1000


def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(dx, dy)
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score(points):
    """Returns (num_exact4, num_5plus, num_3, num_2)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = n5 = n3 = n2 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
        elif c == 3:
            n3 += 1
        elif c == 2:
            n2 += 1
    return n4, n5, n3, n2


def _incremental_score(points, old_pos, new_pos):
    """Fast scoring: only recompute lines involving the moved point."""
    n = len(points)
    # First, get baseline scores from all pairs NOT involving point 0 (the moved one)
    # Actually, we need to be careful. Let's just do a full score but with early exit for validity.
    # For now, keep the same interface but add a fast validity check.
    return _score(points)


def _valid(points):
    """True iff no five are collinear (no line with >=5 points)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= 10:
                return False
    return True


def _valid_fast(points, moved_idx):
    """Fast validity: only check lines through the moved point."""
    n = len(points)
    p = points[moved_idx]
    line_counts = {}
    for j in range(n):
        if j == moved_idx:
            continue
        k = _line_key(p, points[j])
        line_counts[k] = line_counts.get(k, 0) + 1
    # A line through p with j other points has j+1 total points
    # If any has >= 4 others, that's 5 collinear
    for cnt in line_counts.values():
        if cnt >= 4:
            return False
    return True


def _seed_grid():
    return [[x, y] for x in range(4) for y in range(4)]


def _seed_perturbed_grid():
    pts = [[x, y] for x in range(4) for y in range(4)]
    pts[0] = [0, 0]
    pts[3] = [3, 3]
    pts[4] = [0, 5]
    pts[15] = [4, 3]
    return pts


def _seed_3x3_plus():
    pts = [[x, y] for x in range(3) for y in range(3)]
    extra = [(3, 1), (-1, 1), (1, 3), (1, -1), (3, 3), (-1, -1), (4, 0)]
    for e in extra:
        pts.append(list(e))
    return pts


def _seed_lattice():
    base = [[x, y] for x in range(5) for y in range(5)]
    random.shuffle(base)
    return base[:16]


def _seed_diagonal_grid():
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i, j])
    pts[5] = [1, 5]
    pts[10] = [2, -1]
    pts[15] = [5, 3]
    return pts


def _seed_staggered():
    pts = []
    for row in range(4):
        offset = row
        for col in range(4):
            pts.append([col + offset, row])
    return pts


def _seed_hexagonal():
    """A hexagonal-ish arrangement to create more diagonal 4-point lines."""
    pts = []
    # Center and rings
    pts.append([0, 0])
    # Ring 1
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]:
        pts.append([dx, dy])
    # Ring 2 - pick 8 more
    ring2 = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 1), (-2, -1), (1, 2), (-1, -2)]
    for p in ring2:
        pts.append(list(p))
    return pts[:16]


def _seed_affine_grid():
    """Apply an affine transformation to a 4x4 grid to create more 4-point lines."""
    # Use a shear: (x, y) -> (x + y, y)
    pts = []
    for x in range(4):
        for y in range(4):
            pts.append([x + y, y])
    return pts


def _seed_mixed():
    """Mix of grid and staggered."""
    pts = []
    for i in range(4):
        for j in range(4):
            if i % 2 == 0:
                pts.append([i, j])
            else:
                pts.append([i + 1, j])
    return pts


def _local_search(points, rng, time_budget):
    """Single-point local search with incremental scoring."""
    n = len(points)
    cur = [list(p) for p in points]
    cur_n4, cur_n5, cur_n3, cur_n2 = _score(cur)
    best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
    best_pts = [list(p) for p in cur]
    step = 0
    # Track point positions for fast duplicate check
    pos_set = set(map(tuple, cur))
    
    while time.time() < time_budget:
        improved = False
        
        # --- single-point moves ---
        for i in range(n):
            if time.time() >= time_budget:
                break
            x0, y0 = cur[i]
            # Larger adaptive window
            w = 6 + (step // 10)
            w = min(w, 20)
            
            best_local = None
            best_local_key = None
            
            for dx in range(-w, w + 1):
                if time.time() >= time_budget:
                    break
                for dy in range(-w, w + 1):
                    nx, ny = x0 + dx, y0 + dy
                    if abs(nx) > _BOUND or abs(ny) > _BOUND:
                        continue
                    np_ = [nx, ny]
                    if (nx, ny) in pos_set:
                        continue
                    
                    # Fast validity check: only check lines through the moved point
                    # Count how many other points are on each line through np_
                    line_counts = {}
                    for j in range(n):
                        if j == i:
                            continue
                        k = _line_key(np_, cur[j])
                        line_counts[k] = line_counts.get(k, 0) + 1
                    valid = True
                    for cnt in line_counts.values():
                        if cnt >= 4:  # 4 others + moved = 5 collinear
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Quick score: count lines with exactly 4 points
                    # A line has exactly 4 points if it contains the moved point and exactly 3 others
                    # OR it doesn't contain the moved point and had exactly 4 before
                    # We need to be careful: moving point i changes all lines through i
                    # Let's do a targeted score
                    cand = [list(p) for p in cur]
                    cand[i] = np_
                    n4, n5, n3, n2 = _score(cand)
                    key = (n4, -n5, n3, n2)
                    if best_local_key is None or key > best_local_key:
                        best_local_key = key
                        best_local = np_
            
            if best_local is not None:
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if best_local_key > cur_key:
                    pos_set.discard((x0, y0))
                    pos_set.add(tuple(best_local))
                    cur[i] = best_local
                    cur_n4, cur_n5, cur_n3, cur_n2 = best_local_key[0], -best_local_key[1], best_local_key[2], best_local_key[3]
                    improved = True
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]
        
        step += 1
        
        if not improved and step > 2:
            # Random kick to escape local optimum
            if time.time() >= time_budget:
                break
            i = rng.randrange(n)
            x0, y0 = cur[i]
            for _ in range(500):
                nx = x0 + rng.randint(-10, 10)
                ny = y0 + rng.randint(-10, 10)
                if abs(nx) > _BOUND or abs(ny) > _BOUND:
                    continue
                np_ = [nx, ny]
                if (nx, ny) in pos_set:
                    continue
                # Fast validity
                line_counts = {}
                for j in range(n):
                    if j == i:
                        continue
                    k = _line_key(np_, cur[j])
                    line_counts[k] = line_counts.get(k, 0) + 1
                valid = True
                for cnt in line_counts.values():
                    if cnt >= 4:
                        valid = False
                        break
                if not valid:
                    continue
                cand = [list(p) for p in cur]
                cand[i] = np_
                n4, n5, n3, n2 = _score(cand)
                key = (n4, -n5, n3, n2)
                cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                if key >= cur_key:
                    pos_set.discard((x0, y0))
                    pos_set.add((nx, ny))
                    cur[i] = np_
                    cur_n4, cur_n5, cur_n3, cur_n2 = n4, n5, n3, n2
                    if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                        best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                        best_pts = [list(p) for p in cur]
                    break
        
        # Periodic two-point swap for diversity
        if step % 5 == 0 and time.time() < time_budget:
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i != j:
                for _ in range(80):
                    nxi = cur[i][0] + rng.randint(-6, 6)
                    nyi = cur[i][1] + rng.randint(-6, 6)
                    nxj = cur[j][0] + rng.randint(-6, 6)
                    nyj = cur[j][1] + rng.randint(-6, 6)
                    if abs(nxi) > _BOUND or abs(nyi) > _BOUND or abs(nxj) > _BOUND or abs(nyj) > _BOUND:
                        continue
                    if (nxi, nyi) == (nxj, nyj):
                        continue
                    if (nxi, nyi) in pos_set or (nxj, nyj) in pos_set:
                        continue
                    cand = [list(p) for p in cur]
                    cand[i] = [nxi, nyi]
                    cand[j] = [nxj, nyj]
                    if not _valid(cand):
                        continue
                    n4, n5, n3, n2 = _score(cand)
                    key = (n4, -n5, n3, n2)
                    cur_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                    if key > cur_key:
                        pos_set.discard(tuple(cur[i]))
                        pos_set.discard(tuple(cur[j]))
                        pos_set.add((nxi, nyi))
                        pos_set.add((nxj, nyj))
                        cur[i] = [nxi, nyi]
                        cur[j] = [nxj, nyj]
                        cur_n4, cur_n5, cur_n3, cur_n2 = n4, n5, n3, n2
                        if (cur_n4, -cur_n5, cur_n3, cur_n2) > best_key:
                            best_key = (cur_n4, -cur_n5, cur_n3, cur_n2)
                            best_pts = [list(p) for p in cur]
                        break
    
    return (best_key[0], -best_key[1], best_key[2], best_key[3], best_pts)


def construct():
    global _START, _DEADLINE
    rng = random.Random(12345)
    
    seeds = [
        _seed_grid(),
        _seed_perturbed_grid(),
        _seed_3x3_plus(),
        _seed_lattice(),
        _seed_diagonal_grid(),
        _seed_staggered(),
        _seed_hexagonal(),
        _seed_affine_grid(),
        _seed_mixed(),
    ]
    
    # Dedupe
    seen = set()
    uniq = []
    for s in seeds:
        key = tuple(sorted(map(tuple, s)))
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    
    # Add more diverse random seeds
    for _ in range(6):
        pts = []
        for _ in range(16):
            x = rng.randint(-20, 20)
            y = rng.randint(-20, 20)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        while len(pts) < 16:
            x = rng.randint(-30, 30)
            y = rng.randint(-30, 30)
            p = [x, y]
            if p not in pts:
                pts.append(p)
        key = tuple(sorted(map(tuple, pts)))
        if key not in seen:
            seen.add(key)
            uniq.append(pts)
    
    best_overall = None
    per_budget = max(4.0, (_DEADLINE - time.time()) / max(1, len(uniq)))
    
    for s in uniq:
        if time.time() >= _DEADLINE - 2.0:
            break
        if not _valid(s):
            continue
        t_end = min(_DEADLINE - 1.0, time.time() + per_budget)
        res = _local_search(s, rng, t_end)
        if best_overall is None or (res[0], -res[1], res[2], res[3]) > (best_overall[0], -best_overall[1], best_overall[2], best_overall[3]):
            best_overall = res
    
    if best_overall is None:
        return _seed_grid()
    
    pts = best_overall[4]
    
    # Final safety checks
    if not _valid(pts):
        return _seed_grid()
    for p in pts:
        if abs(p[0]) > _BOUND or abs(p[1]) > _BOUND:
            return _seed_grid()
    if len(set(map(tuple, pts))) != 16:
        return _seed_grid()
    
    return [list(p) for p in pts]
