"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import Counter

# EVOLVE-BLOCK-START

def _doubled_area(p, q, r):
    # absolute doubled area (determinant)
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = ax*(by-cy) + bx*(cy-ay) + cx*(ay-by)
    return abs(d)

def _max_multiplicity(pts):
    """Return (max_count, area_value) over positive doubled areas."""
    n = len(pts)
    cnt = Counter()
    for i in range(n):
        pi = pts[i]
        for j in range(i+1, n):
            pj = pts[j]
            for k in range(j+1, n):
                pk = pts[k]
                d = _doubled_area(pi, pj, pk)
                if d > 0:
                    cnt[d] += 1
    if not cnt:
        return (0, 0)
    best_area, best_cnt = max(cnt.items(), key=lambda kv: kv[1])
    return (best_cnt, best_area)

def _valid(pts):
    """Check 30 distinct integer points in [-30,30]^2."""
    if len(pts) != 30:
        return False
    seen = set()
    for (x, y) in pts:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _grid_start():
    # 6 x 5 grid centered
    pts = []
    for x in range(6):
        for y in range(5):
            pts.append((x - 2, y - 2))
    return pts

def _local_search(pts, deadline):
    """Hill-climb: try moving one point to a nearby integer cell."""
    pts = list(pts)
    best_score, _ = _max_multiplicity(pts)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # shuffle order to avoid determinism bias
        order = list(range(30))
        random.shuffle(order)
        for idx in order:
            if time.time() >= deadline:
                break
            x, y = pts[idx]
            # candidate new positions: small neighborhood
            candidates = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if nx == x and ny == y:
                        continue
                    if nx < -30 or nx > 30 or ny < -30 or ny > 30:
                        continue
                    candidates.append((nx, ny))
            random.shuffle(candidates)
            orig = (x, y)
            for (nx, ny) in candidates:
                if time.time() >= deadline:
                    break
                # check collision
                new_pts = list(pts)
                new_pts[idx] = (nx, ny)
                # distinctness
                if len(set(new_pts)) != 30:
                    continue
                score, _ = _max_multiplicity(new_pts)
                if score > best_score:
                    best_score = score
                    pts = new_pts
                    improved = True
                    break
    return pts, best_score

def construct():
    deadline = time.time() + 80.0  # safety margin under 90s
    random.seed(12345)

    best_pts = _grid_start()
    best_score, _ = _max_multiplicity(best_pts)

    # Try several restarts with different starting grids / perturbations
    start_configs = []
    # base grid
    start_configs.append(_grid_start())
    # grid with different aspect
    for (a, b) in [(5, 6), (10, 3), (15, 2), (30, 1)]:
        cfg = []
        for x in range(a):
            for y in range(b):
                cfg.append((x - a // 2, y - b // 2))
        start_configs.append(cfg)
    # random configs
    for _ in range(3):
        cfg = []
        used = set()
        while len(cfg) < 30:
            x = random.randint(-30, 30)
            y = random.randint(-30, 30)
            if (x, y) not in used:
                used.add((x, y))
                cfg.append((x, y))
        start_configs.append(cfg)

    for cfg in start_configs:
        if time.time() >= deadline:
            break
        cfg = list(cfg)
        # small random perturbation to break symmetry
        if random.random() < 0.5:
            for _ in range(random.randint(0, 5)):
                idx = random.randint(0, 29)
                x, y = cfg[idx]
                nx = x + random.randint(-1, 1)
                ny = y + random.randint(-1, 1)
                if -30 <= nx <= 30 and -30 <= ny <= 30 and (nx, ny) not in cfg:
                    cfg[idx] = (nx, ny)
        pts, score = _local_search(cfg, deadline)
        if score > best_score:
            best_score = score
            best_pts = pts
        # continue improving best_pts with more local search
        if time.time() < deadline:
            pts2, score2 = _local_search(best_pts, deadline)
            if score2 > best_score:
                best_score = score2
                best_pts = pts2

    return best_pts

# EVOLVE-BLOCK-END
