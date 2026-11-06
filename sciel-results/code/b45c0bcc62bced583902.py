import time
import random
from collections import Counter
import numpy as np

_START = time.time()
_DEADLINE = 80.0  # safety margin under 90s kill

# Precompute indices for all C(30,3) triples
_TRIPLE_INDICES = None

def _get_triple_indices():
    global _TRIPLE_INDICES
    if _TRIPLE_INDICES is None:
        i, j, k = np.indices((30, 30, 30))
        mask = (i < j) & (j < k)
        _TRIPLE_INDICES = (i[mask].astype(np.int64), j[mask].astype(np.int64), k[mask].astype(np.int64))
    return _TRIPLE_INDICES

def _score(points):
    """Return (max multiplicity, area) over positive doubled areas, or (0,0)."""
    n = len(points)
    if n < 3:
        return (0, 0)
    P = np.array(points, dtype=np.int64)
    I, J, K = _get_triple_indices()
    x1 = P[I, 0]; y1 = P[I, 1]
    x2 = P[J, 0]; y2 = P[J, 1]
    x3 = P[K, 0]; y3 = P[K, 1]
    det = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    nz = det[det > 0]
    if nz.size == 0:
        return (0, 0)
    counts = np.bincount(nz)
    best = int(counts.max())
    best_area = int(np.argmax(counts))
    return (best, best_area)


def _valid(points):
    if len(points) != 30:
        return False
    if len(set(points)) != 30:
        return False
    for (x, y) in points:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return False
    return True


def _line_points(y, offsets):
    """Points on horizontal line y with given x-offsets."""
    return [(x, y) for x in offsets]


def _build_from_lines(lines):
    """lines: list of (y, offsets). Build point set, return None if invalid."""
    pts = []
    for (y, offs) in lines:
        pts.extend(_line_points(y, offs))
    if len(pts) != 30:
        return None
    if len(set(pts)) != 30:
        return None
    for (x, y) in pts:
        if not (-30 <= x <= 30 and -30 <= y <= 30):
            return None
    return pts


def _random_offsets(count, rng, spread=25):
    """Generate `count` distinct random x-offsets in a range, roughly spread."""
    # Use a sorted base plus jitter to encourage spread, then randomize
    base = np.linspace(-spread, spread, count)
    jitter = rng.integers(-3, 4, size=count)
    offs = base + jitter
    # make distinct
    offs = np.sort(offs)
    # dedupe
    uniq = []
    for v in offs:
        if not uniq or v != uniq[-1]:
            uniq.append(int(v))
    # fill if short
    used = set(uniq)
    while len(uniq) < count:
        v = int(rng.integers(-spread, spread + 1))
        if v not in used:
            uniq.append(v)
            used.add(v)
    return uniq[:count]


def _local_search(pts, rng, time_budget):
    """Local search: swap a point with a random candidate to improve score."""
    cur = list(pts)
    cur_score = _score(cur)
    best = list(cur)
    best_score = cur_score
    t0 = time.time()
    # candidate pool: all valid grid points
    pool = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    pool_set = set(pool)
    cur_set = set(cur)
    it = 0
    while time.time() - t0 < time_budget:
        it += 1
        # pick a random index to replace
        i = rng.integers(0, 30)
        # pick a random new point not already in set
        for _ in range(20):
            cand = pool[rng.integers(0, len(pool))]
            if cand not in cur_set:
                break
        else:
            continue
        new_pts = list(cur)
        new_pts[i] = cand
        ns, na = _score(new_pts)
        if (ns, na) > best_score:
            best = new_pts
            best_score = (ns, na)
            cur = new_pts
            cur_set = set(cur)
            cur_score = (ns, na)
        # occasional random restart-ish: if stuck, try a different point
    return best, best_score


def _seed_grid():
    # 6x5 grid baseline
    return [(x, y) for x in range(6) for y in range(5)]


def construct():
    global _START
    rng = random.Random(12345)
    best = _seed_grid()
    best_score = _score(best)

    # Phase 1: structured line-based constructions
    # Try various numbers of lines and spacings
    line_configs = []
    # 3 lines
    for y0 in range(-20, 21, 5):
        for dy in range(3, 20):
            ys = [y0, y0 + dy, y0 + 2 * dy]
            if max(ys) > 30 or min(ys) < -30:
                continue
            # distribute 30 points: e.g. 10/10/10
            for (a, b, c) in [(10, 10, 10), (11, 10, 9), (9, 10, 11), (12, 9, 9)]:
                line_configs.append((ys, [a, b, c]))
    # 4 lines
    for y0 in range(-15, 16, 4):
        for dy in range(3, 15):
            ys = [y0, y0 + dy, y0 + 2 * dy, y0 + 3 * dy]
            if max(ys) > 30 or min(ys) < -30:
                continue
            for (a, b, c, d) in [(8, 8, 7, 7), (7, 8, 8, 7), (9, 7, 8, 6)]:
                line_configs.append((ys, [a, b, c, d]))

    # Evaluate a sample of line configs with random offsets
    t0 = time.time()
    sampled = 0
    for (ys, counts) in line_configs:
        if time.time() > _DEADLINE - 30:
            break
        for trial in range(2):
            if time.time() > _DEADLINE - 30:
                break
            lines = []
            ok = True
            for (y, cnt) in zip(ys, counts):
                offs = _random_offsets(cnt, rng, spread=28)
                lines.append((y, offs))
            pts = _build_from_lines(lines)
            if pts is None:
                continue
            sampled += 1
            sc, sa = _score(pts)
            if (sc, sa) > best_score:
                best_score = (sc, sa)
                best = pts
                # local polish
                if time.time() < _DEADLINE - 20:
                    lp, lsc = _local_search(best, rng, time_budget=min(3.0, _DEADLINE - 20 - time.time()))
                    if lsc > best_score:
                        best_score = lsc
                        best = lp

    # Phase 2: local search from best and from seed
    if time.time() < _DEADLINE - 10:
        budget = min(15.0, _DEADLINE - 10 - time.time())
        lp, lsc = _local_search(best, rng, time_budget=budget)
        if lsc > best_score:
            best_score = lsc
            best = lp

    if time.time() < _DEADLINE - 5:
        budget = min(10.0, _DEADLINE - 5 - time.time())
        lp2, lsc2 = _local_search(_seed_grid(), rng, time_budget=budget)
        if lsc2 > best_score:
            best_score = lsc2
            best = lp2

    # Final validity check
    if not _valid(best):
        best = _seed_grid()
    return best


# Pre-run once at import to warm up and cache? No—construct is called by evaluator.
# Ensure a baseline is always valid.
_construct_baseline = _seed_grid()
