"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_LIMIT = 1000
_N = 16
_DEADLINE = 140.0  # safety margin vs 150s wall-clock kill


def _build_pool():
    pts = []
    seen = set()
    for x in range(-30, 31):
        for y in range(-30, 31):
            seen.add((x, y))
    pts = list(seen)
    rng = random.Random(12345)
    extra = 0
    while extra < 4000:
        x = rng.randint(-1000, 1000)
        y = rng.randint(-1000, 1000)
        if (x, y) not in seen:
            seen.add((x, y))
            pts.append((x, y))
            extra += 1
    return pts


_POOL = _build_pool()


def _slope_key(xi, yi, xj, yj):
    """Reduced integer direction (dx,dy) with canonical sign."""
    dx = xj - xi
    dy = yj - yi
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    dx //= g
    dy //= g
    # canonical orientation
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy)


def _count_lines(pts):
    """pts: list of (x,y). Return (num_exact_4, num_ge5).

    O(n^2): for each point, group other points by reduced slope; a slope
    seen at >=3 points through i (i.e., group size >=2 others => >=3 total
    on that line) is a line. We collect unique lines via (slope, min-member).
    """
    n = len(pts)
    if n < 4:
        return 0, 0
    lines = {}
    for i in range(n):
        xi, yi = pts[i]
        groups = {}
        for j in range(n):
            if i == j:
                continue
            k = _slope_key(xi, yi, pts[j][0], pts[j][1])
            if k is None:
                continue
            groups.setdefault(k, []).append(j)
        for k, idxs in groups.items():
            if len(idxs) >= 2:  # at least 3 points on this line (incl i)
                members = sorted([i] + idxs)
                key = (k, members[0])
                lines.setdefault(key, set()).update(members)
    exact4 = 0
    ge5 = 0
    for key, mem in lines.items():
        m = len(mem)
        if m == 4:
            exact4 += 1
        elif m >= 5:
            ge5 += 1
    return exact4, ge5


def _score(pts_list):
    e4, ge5 = _count_lines(pts_list)
    return e4 - 1000 * ge5


def _pts_list(pts_set):
    return list(pts_set)


def _random_start(rng):
    pts = set()
    while len(pts) < _N:
        pts.add(rng.choice(_POOL))
    return pts


def _neighbor(pts_set, rng):
    pts = list(pts_set)
    op = rng.random()
    if op < 0.60:
        i = rng.randrange(len(pts))
        pts[i] = rng.choice(_POOL)
    else:
        i, j = rng.sample(range(len(pts)), 2)
        pts[i] = rng.choice(_POOL)
        pts[j] = rng.choice(_POOL)
    newset = set()
    for p in pts:
        if p not in newset:
            newset.add(p)
    while len(newset) < _N:
        newset.add(rng.choice(_POOL))
    return newset


def _local_polish(pts_set, rng, deadline):
    best = set(pts_set)
    best_s = _score(_pts_list(best))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        bl = list(best)
        for i in range(_N):
            if time.time() >= deadline:
                break
            old = bl[i]
            for _ in range(6):
                cand = rng.choice(_POOL)
                if cand == old:
                    continue
                trial = list(bl)
                trial[i] = cand
                s = _score(trial)
                if s > best_s:
                    best = set(trial)
                    best_s = s
                    bl = list(best)
                    improved = True
                    break
    return best, best_s


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(20240517)

    grid = {(x, y) for x in range(4) for y in range(4)}
    best = grid
    best_s = _score(_pts_list(grid))  # expect 10

    t0 = 1.0
    tmin = 0.001
    cooling = 0.985
    cur = _random_start(rng)
    cur_s = _score(_pts_list(cur))
    if cur_s > best_s:
        best = set(cur)
        best_s = cur_s

    step = 0
    while time.time() < deadline - 8.0:
        if step % 300 == 0:
            cur = _random_start(rng)
            cur_s = _score(_pts_list(cur))
            if cur_s > best_s:
                best = set(cur)
                best_s = cur_s
        trial = _neighbor(cur, rng)
        trial_s = _score(_pts_list(trial))
        delta = trial_s - cur_s
        if delta >= 0:
            cur = trial
            cur_s = trial_s
            if cur_s > best_s:
                best = set(cur)
                best_s = cur_s
        else:
            if t0 > tmin:
                if rng.random() < delta / (t0 * 10.0 + 1e-9):
                    cur = trial
                    cur_s = trial_s
        t0 *= cooling
        step += 1

    polished, ps = _local_polish(best, rng, deadline - 0.5)
    if ps > best_s:
        best = polished
        best_s = ps

    out = [[x, y] for (x, y) in best]
    if len(out) != _N:
        out = [[x, y] for (x, y) in grid]
    return out
# EVOLVE-BLOCK-END
