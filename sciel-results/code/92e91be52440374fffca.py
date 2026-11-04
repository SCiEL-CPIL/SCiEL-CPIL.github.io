"""Fresh design: soft-min surrogate + L-BFGS-B polish over diverse starts, with deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_START = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0  # seconds before kill to stop

# Precompute triangle index pairs once
def _make_pairs(n):
    idx = np.arange(n)
    ii, jj, kk = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (ii < jj) & (jj < kk)
    return ii[mask], jj[mask], kk[mask]

_I, _J, _K = _make_pairs(N)


def _hull_area(pts):
    try:
        h = ConvexHull(pts)
        return h.volume  # area in 2D
    except Exception:
        return 1.0


def _min_tri_area(pts):
    # area = 0.5 * |cross|
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    return float(areas.min())


def _score(pts):
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 0.0
    return _min_tri_area(pts) / ha


def _soft_min_obj(x, beta):
    pts = x.reshape(-1, 2)
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    ha = _hull_area(pts)
    if ha <= 1e-12:
        return 1e6
    # soft-min of (areas / ha); maximize => minimize negative
    m = areas.min()
    # stable logsumexp of areas
    shifted = areas - m
    lse = m + np.log(np.sum(np.exp(shifted)))
    soft_min = lse / ha
    return -soft_min


def _soft_min_grad(x, beta):
    # Numerical gradient is slow; use analytical via finite diff of a few coords is heavy.
    # We'll rely on L-BFGS-B numerical gradient but that's O(n) evals per step.
    # Instead, provide None and let L-BFGS-B do numerical gradient (acceptable for 28 vars).
    return None


def _polish(x0, beta):
    x0 = np.asarray(x0, dtype=float)
    res = minimize(_soft_min_obj, x0, args=(beta,), method='L-BFGS-B',
                   jac=None, options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-9})
    return res.x


def _local_refine(pts):
    # Coordinate-wise hill climbing on the true (non-smooth) score to escape surrogate bias.
    pts = np.array(pts, dtype=float)
    best = _score(pts)
    step = 0.05
    improved = True
    iters = 0
    while improved and iters < 40:
        improved = False
        iters += 1
        for i in range(N):
            for d in range(2):
                for sign in (1, -1):
                    cand = pts.copy()
                    cand[i, d] += sign * step
                    s = _score(cand)
                    if s > best:
                        best = s
                        pts = cand
                        improved = True
    return pts, best


def _regular_polygon(n, r=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


def _jittered_polygon(n, r=1.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.1, 0.1, n)
    rr = r * (1 + rng.uniform(-0.15, 0.15, n))
    return np.column_stack([rr * np.cos(t), rr * np.sin(t)])


def _spherical_like(n, seed=0):
    # Fibonacci sphere projected to plane, scaled
    rng = np.random.RandomState(seed)
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i + rng.uniform(-0.05, 0.05, n)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    pts = np.column_stack([x, y])
    # scale to unit-ish
    pts *= 1.5
    return pts


def _grid_start(n, seed=0):
    rng = np.random.RandomState(seed)
    # 4x4 grid perturbed
    base = np.array([[i % 4, i // 4] for i in range(n)], dtype=float)
    base = base * 0.5 + 0.25
    base = base + rng.uniform(-0.1, 0.1, base.shape)
    return base


def construct():
    global _START
    t0 = time.time()
    # Candidate starts
    starts = []
    starts.append(_regular_polygon(N, r=1.0))
    for s in range(6):
        starts.append(_jittered_polygon(N, r=1.0, seed=s))
    for s in range(4):
        starts.append(_spherical_like(N, seed=s))
    for s in range(4):
        starts.append(_grid_start(N, seed=s))

    best_pts = None
    best_score = -1.0

    # Stage 1: coarse polish (low beta) on all starts
    for x0 in starts:
        if time.time() - _START > _DEADLINE - _SAFETY:
            break
        x1 = _polish(x0, beta=50)
        s = _score(x1)
        if s > best_score:
            best_score = s
            best_pts = x1

    # Stage 2: sharp polish (high beta) on top starts
    if best_pts is not None:
        if time.time() - _START < _DEADLINE - _SAFETY:
            x2 = _polish(best_pts, beta=250)
            s2 = _score(x2)
            if s2 > best_score:
                best_score = s2
                best_pts = x2

    # Local coordinate refinement on the true objective
    if best_pts is not None and time.time() - _START < _DEADLINE - _SAFETY:
        rp, rs = _local_refine(best_pts)
        if rs > best_score:
            best_score = rs
            best_pts = rp

    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N, r=1.0)
        best_score = _score(best_pts)

    return np.asarray(best_pts, dtype=float)


if __name__ == "__main__":
    p = construct()
    print("score:", _score(p))
