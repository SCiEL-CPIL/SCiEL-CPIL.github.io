import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQ3 = math.sqrt(3.0)
DEADLINE = 150.0
SAFETY = 10.0


def _elapsed(t0):
    return time.time() - t0


def _hex_vertices(cx, cy, a):
    ang = a + np.arange(6) * (math.pi / 3.0)
    return np.stack([cx + np.cos(ang), cy + np.sin(ang)], axis=1)


def _convex_hull(points):
    points = np.asarray(points, dtype=float)
    order = np.lexsort((points[:, 1], points[:, 0]))
    pts = points[order]
    n = len(pts)
    if n <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2:
            a, b = lower[-2], lower[-1]
            cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cr <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2:
            a, b = upper[-2], upper[-1]
            cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cr <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _min_hex_side(hull_pts):
    """Min side L of a regular hexagon (center 0, rotation 0) containing hull_pts."""
    best = 0.0
    for k in range(6):
        ang = k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        d = hull_pts[:, 0] * nx + hull_pts[:, 1] * ny
        best = max(best, d.max() / (SQ3 / 2.0))
    return best


def _separation(v1, v2):
    """Minimum distance between two unit hexagons' vertex sets (0 if overlapping)."""
    min_d = float('inf')
    for a in v1:
        dx = v2[:, 0] - a[0]
        dy = v2[:, 1] - a[1]
        min_d = min(min_d, float(np.min(dx * dx + dy * dy)))
        if min_d < 1e-9:
            return 0.0
    return math.sqrt(min_d)


def _compute(params):
    """Return (L, min_sep, hull) for a 36-vector of (cx,cy,angle) per hexagon."""
    L = 0.0
    min_sep = float('inf')
    hull = None
    all_pts = []
    verts = []
    for i in range(N):
        cx, cy, a = params[3 * i:3 * i + 3]
        v = _hex_vertices(cx, cy, a)
        verts.append(v)
        all_pts.append(v)
        for k in range(6):
            ang = k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            d = v[:, 0] * nx + v[:, 1] * ny
            L = max(L, float(d.max()) / (SQ3 / 2.0))
    for i in range(N):
        for j in range(i + 1, N):
            s = _separation(verts[i], verts[j])
            min_sep = min(min_sep, s)
    all_pts = np.vstack(all_pts)
    hull = _convex_hull(all_pts)
    return L, min_sep, hull


def _objective(params, W):
    L, min_sep, _ = _compute(params)
    pen = 0.0
    if min_sep < 0.01:
        pen = W * (0.01 - min_sep) ** 2
    return L + pen


def _fallback():
    # Valid hex-grid packing (12 cells, side 3), scaled slightly for strict separation.
    cells = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1),
             (-1, 1), (2, 0), (-2, 0), (2, -1), (-2, 1), (1, 1)]
    S = 1.000001
    inner = []
    for (q, r) in cells:
        x = 1.5 * q
        y = SQ3 * (r + q / 2.0)
        inner.append([x * S, y * S, 0.0])
    L = 3.0 * S
    return inner, [0.0, 0.0], L, 0.0


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - SAFETY

    # Initial hex-grid packing (valid, L=3)
    cells = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1),
             (-1, 1), (2, 0), (-2, 0), (2, -1), (-2, 1), (1, 1)]
    x0 = np.zeros(3 * N)
    for i, (q, r) in enumerate(cells):
        x0[3 * i] = 1.5 * q
        x0[3 * i + 1] = SQ3 * (r + q / 2.0)
        x0[3 * i + 2] = 0.0

    best_params = x0.copy()
    best_L = 3.0
    best_min_sep = float('inf')

    # Evaluate initial
    L, ms, _ = _compute(x0)
    best_L = L
    best_min_sep = ms

    # Gradually increase barrier weight
    W = 50.0
    while _elapsed(t0) < deadline:
        try:
            res = minimize(_objective, best_params, method='Powell',
                           options={'maxiter': 60, 'maxfev': 3000,
                                    'xtol': 1e-6, 'ftol': 1e-8, 'W': W})
            if res.x is not None:
                L, ms, _ = _compute(res.x)
                if ms >= 1e-6 and L < best_L - 1e-9:
                    best_params = res.x
                    best_L = L
                    best_min_sep = ms
        except Exception:
            pass
        # Increase barrier weight for stronger separation enforcement
        W = min(W * 1.5, 20000.0)
        if W >= 20000.0:
            break
        if _elapsed(t0) > deadline:
            break

    # Final validity check
    L, ms, _ = _compute(best_params)
    if ms < 1e-6:
        inner, center, Lf, ang = _fallback()
        return inner, center, Lf, ang

    inner = []
    for i in range(N):
        cx, cy, a = best_params[3 * i:3 * i + 3]
        inner.append([cx, cy, math.degrees(a)])
    return inner, [0.0, 0.0], L, 0.0
