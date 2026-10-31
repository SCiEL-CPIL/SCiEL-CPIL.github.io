import time
import math
import random
import numpy as np

N = 11
_SEP = 2.0          # min center distance for unit hexagons (same orientation)
_EPS = 1e-7
_DEADLINE = 135.0   # safety margin under 150s wall clock


def _hex_radius(side, theta):
    """Max distance from center to boundary of a flat-top hexagon of given side,
    at direction theta (radians)."""
    side = float(side)
    if side <= 0:
        return 0.0
    t = theta % (math.pi / 3.0)
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    return side * cos_t / (cos_t + math.sqrt(3.0) * sin_t)


def _inside_eroded(x, y, L):
    """True if a unit hexagon centered at (x,y) fits inside outer hexagon (center 0, side L)."""
    if L <= 1.0 + _EPS:
        return False
    r = math.hypot(x, y)
    if r < _EPS:
        return True
    return r <= _hex_radius(L - 1.0, math.atan2(y, x)) + _EPS


def _greedy(centers, L, seed):
    rng = random.Random(seed)
    chosen = []
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    n = len(centers)
    for _ in range(N):
        d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
        if chosen:
            ok = np.array([_inside_eroded(c[0], c[1], L) for c in centers])
            for i in range(n):
                if not ok[i]:
                    continue
                if d[i, chosen].min() < _SEP - _EPS:
                    ok[i] = False
            if not ok.any():
                return None
            idx = [i for i in range(n) if ok[i]]
            i = idx[rng.randrange(len(idx))]
        else:
            ok = np.array([_inside_eroded(c[0], c[1], L) for c in centers])
            if not ok.any():
                return None
            idx = [i for i in range(n) if ok[i]]
            i = idx[rng.randrange(len(idx))]
        chosen.append(i)
    return [centers[i] for i in chosen]


def _local_improve(centers, L, seed, iters=300):
    rng = random.Random(seed)
    pts = np.array(centers, dtype=float)
    n = len(pts)
    # penalty: separation violations + containment violations
    def score(p):
        d = np.hypot(p[:, None] - p[None, :], p[:, None, 1] - p[:, None, 1])
        np.fill_diagonal(d, 1e9)
        sep_pen = np.sum(np.maximum(_SEP - d, 0.0) ** 2)
        con_pen = 0.0
        for i in range(n):
            r = math.hypot(p[i, 0], p[i, 1])
            if r > _EPS:
                lim = _hex_radius(L - 1.0, math.atan2(p[i, 1], p[i, 0]))
                con_pen += max(0.0, r - lim) ** 2
        return sep_pen + 10.0 * con_pen
    cur = score(pts)
    base = np.array(pts, dtype=float)
    for _ in range(iters):
        i = rng.randrange(n)
        new = pts.copy()
        ang = rng.random() * 2.0 * math.pi
        step = rng.uniform(0.05, 0.6)
        new[i, 0] += step * math.cos(ang)
        new[i, 1] += step * math.sin(ang)
        s = score(new)
        if s <= cur + 1e-12:
            pts = new
            cur = s
    return pts, cur


def _solve_for_L(L, start_seed):
    # candidate centers: hexagonal lattice points within eroded region
    s = L - 1.0
    R = s * 1.05
    pts = []
    if s > _EPS:
        a = _SEP / 2.0          # lattice spacing for unit hexagons
        b = a * math.sqrt(3.0)
        nmax = int(R / a) + 2
        for i in range(-nmax, nmax + 1):
            for j in range(-nmax, nmax + 1):
                x = i * a
                y = j * b + (i % 2) * b / 2.0
                if _inside_eroded(x, y, L):
                    pts.append((x, y))
    # also add a fine radial/grid set
    ng = 40
    for gx in range(-ng, ng + 1):
        for gy in range(-ng, ng + 1):
            x = gx * (2.0 * s) / ng
            y = gy * (2.0 * s) / ng
            if _inside_eroded(x, y, L):
                pts.append((x, y))
    if not pts:
        return None
    best = None
    seed = start_seed
    for attempt in range(400):
        cand = _greedy(pts, L, seed)
        if cand is None:
            seed += 1
            continue
        p, sc = _local_improve(cand, L, seed, iters=150)
        ok = True
        for i in range(N):
            if not _inside_eroded(p[i, 0], p[i, 1], L):
                ok = False
                break
        if ok:
            d = np.hypot(p[:, None] - p[None, :], p[:, None, 1] - p[:, None, 1])
            np.fill_diagonal(d, 1e9)
            if d.min() >= _SEP - 1e-6:
                if best is None or sc < best[1]:
                    best = (p, sc)
        seed += 1
    return best


def _valid(p, L, tol=1e-5):
    for i in range(len(p)):
        if not _inside_eroded(p[i, 0], p[i, 1], L):
            return False
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if math.hypot(p[i, 0] - p[j, 0], p[i, 1] - p[j, 1]) < _SEP - tol:
                return False
    return True


def construct():
    t0 = time.time()
    # seed fallback (collinear, side 40)
    seed_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    best = (seed_inner, (0.0, 0.0), 40.0, 0.0)

    def elapsed():
        return time.time() - t0

    # coarse scan to find feasible region
    lo, hi = 3.0, 40.0
    # verify hi is feasible
    if not _valid(seed_inner, 40.0):
        hi = 60.0
    # find a feasible hi
    feasible_hi = None
    for L in [12.0, 10.0, 9.0, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0]:
        if elapsed() > _DEADLINE - 20:
            break
        res = _solve_for_L(L, 12345)
        if res is not None and _valid(res[0], L):
            feasible_hi = L
            best = ([list(map(float, res[0][i])) + [0.0] for i in range(N)], (0.0, 0.0), L, 0.0)
            break
    if feasible_hi is None:
        # use seed
        return best

    lo = max(3.0, feasible_hi - 1.5)
    hi = feasible_hi

    # binary search
    for _ in range(14):
        if elapsed() > _DEADLINE - 15:
            break
        mid = 0.5 * (lo + hi)
        res = _solve_for_L(mid, 99991)
        if res is not None and _valid(res[0], mid):
            hi = mid
            best = ([list(map(float, res[0][i])) + [0.0] for i in range(N)], (0.0, 0.0), mid, 0.0)
        else:
            lo = mid

    # final polish at hi
    if elapsed() < _DEADLINE - 10:
        res = _solve_for_L(hi, 77777)
        if res is not None and _valid(res[0], hi):
            best = ([list(map(float, res[0][i])) + [0.0] for i in range(N)], (0.0, 0.0), hi, 0.0)

    return best
