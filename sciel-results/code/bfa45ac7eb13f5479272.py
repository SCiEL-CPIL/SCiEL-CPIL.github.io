# EVOLVE-BLOCK-START
import time
import numpy as np

# ----------------- fixed problem data -----------------
T0 = 630.0          # hard wall-clock limit
MARGIN = 20.0       # safety margin before limit
SOFA_X = np.arange(-3.0, 1.5, 0.01)
SOFA_Y = np.arange(-1.5, 1.5, 0.01)
GX, GY = np.meshgrid(SOFA_X, SOFA_Y, indexing="xy")
GX = GX.ravel()
GY = GY.ravel()
NPTS = GX.size
# grid points of the base corridor (inner corner at origin, a=0)
BX = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 0.0, -1.0, -2.0, -3.0])
BY = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
# outward normals (unit) and offsets of the 8 base half-planes
BASE_N = np.array([
    [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
    [-1.0, -1.0] / np.sqrt(2.0), [1.0, -1.0] / np.sqrt(2.0),
    [1.0, 1.0] / np.sqrt(2.0), [0.0, -1.0], [-1.0, 1.0] / np.sqrt(2.0),
])
BASE_C = np.array([3.0, 1.0, 1.0, 3.0 * np.sqrt(2.0), -3.0 * np.sqrt(2.0),
                   3.0 * np.sqrt(2.0), -1.0, -3.0 * np.sqrt(2.0)])


def _deadline_ok(t0):
    return (t0 + MARGIN) - time.time() > 0


def _rot(a):
    r = np.deg2rad(a)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s], [s, c]])


def _build_path(rows):
    rows = np.asarray(rows, dtype=np.float64)
    full = np.vstack([[[0.0, 0.0, 0.0]], rows, [[-2.0, 0.0, 90.0]]])
    full = np.concatenate([full, np.zeros((0, 3))], axis=0)
    # densify: linear interp of (x,y,a) in m = max(1, ceil(|a2-a1|)) steps
    out = [full[0]]
    for i in range(len(full) - 1):
        p0, p1 = full[i], full[i + 1]
        m = max(1, int(np.ceil(abs(p1[2] - p0[2]))))
        for k in range(1, m + 1):
            t = k / m
            out.append(p0 + t * (p1 - p0))
    return np.array(out)


def _halfplanes(path):
    """Return (N,2) normals and (N,) offsets for all placements in path."""
    n = len(path)
    N = np.empty((n * 8, 2))
    C = np.empty(n * 8)
    for i in range(n):
        x, y, a = path[i]
        R = _rot(a)
        c, s = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
        # point on boundary: p = (x,y) + R @ (base_pt - (x,y))
        # inward half-plane n . w <= c, n = R @ base_n, c = n . p
        nmat = R @ BASE_N.T                       # (2,8)
        pts = (x + c * BX - s * BY, y + s * BX + c * BY)  # (2,8)
        nmat = nmat.T                             # (8,2)
        pmat = np.stack(pts, axis=1)              # (8,2)
        N[i * 8:(i + 1) * 8] = nmat
        C[i * 8:(i + 1) * 8] = pmat @ nmat.T
    return N, C


def _area_fast(N, C):
    """Fast concave area estimate of intersection of half-planes n.w <= c over grid."""
    d = GX * N[:, 0] + GY * N[:, 1] - C          # (M, N)
    # soft-min of the (M,) values  max_j (n_j . w - c_j)
    # use log-sum-exp with temperature
    temp = 0.05
    m = d.max(axis=1, keepdims=True)
    z = np.exp((d - m) / temp)
    h = m + temp * np.log(z.sum(axis=1))         # (M,) approx max
    # inside if h <= 0 ; area = count * 1e-4
    return float(np.sum(h <= 0.0) * 1e-4)


def _area_exact(rows):
    """Exact count of grid points inside every placement (boundary included)."""
    path = _build_path(rows)
    n = len(path)
    inside = np.ones(NPTS, dtype=bool)
    for i in range(n):
        x, y, a = path[i]
        r = np.deg2rad(a)
        c, s = np.cos(r), np.sin(r)
        u = c * (GX - x) + s * (GY - y)
        v = -s * (GX - x) + c * (GY - y)
        horiz = (u >= -3.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        inside &= (horiz | vert)
        if not inside.any():
            break
    return float(np.sum(inside) * 1e-4)


def _grad_area(rows):
    """Gradient of fast area w.r.t. (x,y,a) of each interior row (central diff)."""
    rows = np.array(rows, dtype=np.float64, copy=True)
    n = len(rows)
    base = _area_fast(*_halfplanes(_build_path(rows)))
    g = np.zeros((n, 3))
    eps = 0.02
    for i in range(n):
        for j in range(3):
            rows[i, j] += eps
            ap = _area_fast(*_halfplanes(_build_path(rows)))
            rows[i, j] -= 2 * eps
            am = _area_fast(*_halfplanes(_build_path(rows)))
            rows[i, j] += eps
            g[i, j] = (ap - am) / (2 * eps)
    return base, g


def _polish(rows, t0, iters=60, lr=0.05):
    """Local ascent on fast area using subgradient (concave -> any ascent works)."""
    rows = np.array(rows, dtype=np.float64, copy=True)
    best = _area_exact(rows)
    best_rows = rows.copy()
    for it in range(iters):
        if not _deadline_ok(t0):
            break
        base, g = _grad_area(rows)
        if base > best:
            best = base
            best_rows = rows.copy()
        # normalize gradient
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-9:
            break
        step = lr * g / gnorm
        rows = rows + step
        # keep within sensible bounds
        rows[:, 0] = np.clip(rows[:, 0], -2.5, 0.5)
        rows[:, 1] = np.clip(rows[:, 1], -0.5, 1.5)
        rows[:, 2] = np.clip(rows[:, 2], 1.0, 89.0)
    # final exact polish
    exact = _area_exact(best_rows)
    return best_rows, exact


def moving_sofa_2d() -> np.ndarray:
    t0 = time.time()
    # hand-set baseline path (8 interior rows)
    rows = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)

    best_rows = rows.copy()
    best_score = _area_exact(best_rows)

    # local polish via subgradient ascent on fast surrogate
    if _deadline_ok(t0):
        polished, score = _polish(rows, t0)
        if score > best_score:
            best_rows = polished
            best_score = score

    # a few extra restarts if time remains
    rng = np.random.default_rng(12345)
    while _deadline_ok(t0) and (t0 + MARGIN - time.time()) > 30.0:
        cand = best_rows + rng.normal(0.0, 0.05, size=best_rows.shape)
        cand[:, 0] = np.clip(cand[:, 0], -2.5, 0.5)
        cand[:, 1] = np.clip(cand[:, 1], -0.5, 1.5)
        cand[:, 2] = np.clip(cand[:, 2], 1.0, 89.0)
        polished, score = _polish(cand, t0, iters=25, lr=0.05)
        if score > best_score:
            best_rows = polished
            best_score = score

    return best_rows


# EVOLVE-BLOCK-END
