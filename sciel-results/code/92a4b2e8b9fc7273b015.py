# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time


def _min_triangle_area(pts):
    """Minimum area among all C(n,3) triangles, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    # Build index arrays for all (i,j,k), i<j<k
    idx = np.array([[i, j, k] for i in range(n)
                    for j in range(i + 1, n)
                    for k in range(j + 1, n)], dtype=int)
    a = pts[idx[:, 0]]  # (286, 2)
    b = pts[idx[:, 1]]
    c = pts[idx[:, 2]]
    # area = 0.5*|x1(y2-y3)+x2(y3-y1)+x3(y1-y2)|
    area = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1]) +
                        b[:, 0] * (c[:, 1] - a[:, 1]) +
                        c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(area.min())


def _hull_area(pts):
    """Convex hull area; 0 on failure."""
    try:
        return float(ConvexHull(pts).volume)
    except Exception:
        return 0.0


def _score(pts):
    """A_min / A_hull, or 0 if invalid."""
    if pts.shape != (13, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha <= 0 or not np.isfinite(ha):
        return 0.0
    ma = _min_triangle_area(pts)
    if ma <= 0:
        return 0.0
    return ma / ha


def _regular_polygon(n=13):
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return np.vstack([pts, [[0.0, 0.0]]])


def _halton(n=13):
    def halton(i, base):
        r, f = 0.0, 1.0 / base
        while i > 0:
            r += f * (i % base)
            i //= base
            f /= base
        return r
    pts = np.array([[halton(i + 1, 2), halton(i + 1, 3)] for i in range(n)])
    pts = (pts - pts.mean(axis=0)) * 2.0
    return pts


def _grid(n=13):
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    g = np.array([[x, y] for x in xs for y in ys])
    return g[:n]


def _random(n=13, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def _simulated_annealing(pts, time_limit, rng):
    """Simulated annealing maximizing A_min/A_hull."""
    cur = pts.astype(float).copy()
    cur_score = _score(cur)
    best = cur.copy()
    best_score = cur_score
    if cur_score <= 0:
        return best, best_score
    start = time.time()
    T = 0.05
    cool = 0.9995
    n = len(cur)
    while time.time() - start < time_limit:
        # perturb a random subset of points
        nmove = rng.integers(1, 4)
        idx = rng.choice(n, size=nmove, replace=False)
        cur[idx] += T * rng.standard_normal((nmove, 2))
        new_score = _score(cur)
        if new_score > cur_score:
            cur_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = cur.copy()
        else:
            # accept downhill with probability exp((new-cur)/T)
            if rng.random() < np.exp((new_score - cur_score) / max(T, 1e-12)):
                cur_score = new_score
        T *= cool
        if T < 1e-7:
            break
    return best, best_score


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin under 630s wall-clock
    best_points = None
    best_score = 0.0
    rng = np.random.default_rng(0)

    inits = [_regular_polygon(), _halton(), _grid()]
    for s in range(1, 9):
        inits.append(_random(13, seed=s))

    for init in inits:
        if time.time() > deadline - 30:
            break
        if not np.all(np.isfinite(init)):
            continue
        if _hull_area(init) <= 0:
            continue
        time_remaining = deadline - time.time() - 25
        if time_remaining <= 0:
            break
        opt, sc = _simulated_annealing(init, time_limit=min(time_remaining, 60.0), rng=rng)
        if sc > best_score:
            best_score = sc
            best_points = opt.copy()

    if best_points is None or not np.all(np.isfinite(best_points)) or best_score <= 0:
        best_points = _regular_polygon()

    return best_points


# EVOLVE-BLOCK-END
