import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQ3 = math.sqrt(3.0)
DEADLINE = 143.0

# Unit regular hexagon (side 1), vertices in CCW order, flat-top orientation.
# apothem = SQ3/2, circumradius = 1.
_HEX = np.array([
    [1.0, 0.0],
    [0.5, SQ3 / 2.0],
    [-0.5, SQ3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -SQ3 / 2.0],
    [0.5, -SQ3 / 2.0],
], dtype=float)
# Edge normals (outward), matching _HEX edge order.
_NRM = np.array([
    [SQ3 / 2.0, 0.5],
    [0.0, 1.0],
    [-SQ3 / 2.0, 0.5],
    [-SQ3 / 2.0, -0.5],
    [0.0, -1.0],
    [SQ3 / 2.0, -0.5],
], dtype=float)
AP = SQ3 / 2.0  # apothem of unit hexagon
N_PAIRS = N * (N - 1) // 2
PAIR_IDX = [(i, j) for i in range(N) for j in range(i + 1, N)]


def _rot(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _build_rot(angles):
    return [_rot(a) for a in angles]


def _pair_sep(verts, R, centers):
    """Minimum distance between two convex hexagons (support-function)."""
    best = -1e9
    for a in range(6):
        n = R[0] @ _NRM[a]
        h1 = float(np.max(verts[0] @ n))
        h2 = float(np.max(verts[1] @ n))
        d = h1 - h2
        if d > best:
            best = d
        n = -n
        h1 = float(np.max(verts[0] @ n))
        h2 = float(np.max(verts[1] @ n))
        d = h1 - h2
        if d > best:
            best = d
    if best >= -1e-9:
        return 0.0
    return -best


def _all_seps(centers, angles):
    R = _build_rot(angles)
    verts = [centers[i] + _HEX @ R[i].T for i in range(N)]
    seps = np.empty(N_PAIRS)
    for k, (i, j) in enumerate(PAIR_IDX):
        seps[k] = _pair_sep(verts[i], R, centers)
    return seps


def _outer_side(centers, angles, outer_ang):
    """Smallest outer regular-hexagon side enclosing all inner hexagons."""
    R = _build_rot(angles)
    verts = [centers[i] + _HEX @ R[i].T for i in range(N)]
    best = 1e9
    base = outer_ang
    for k in range(6):
        n = _rot(base + k * math.pi / 3) @ _NRM[0]
        n = n / np.linalg.norm(n)
        maxp = max(float(np.max(v @ n)) for v in verts)
        maxm = max(float(np.max(v @ (-n))) for v in verts)
        S = (maxp + maxm) / AP
        if S < best:
            best = S
    return best


def _objective(x, outer_ang):
    centers = x[: 2 * N].reshape(N, 2)
    angles = x[2 * N:]
    return _outer_side(centers, angles, outer_ang)


def _objective_with_pen(x, outer_ang, min_sep):
    centers = x[: 2 * N].reshape(N, 2)
    angles = x[2 * N:]
    S = _outer_side(centers, angles, outer_ang)
    seps = _all_seps(centers, angles)
    pen = 0.0
    for s in seps:
        if s < min_sep:
            pen += (min_sep - s) ** 2
    return S + 1e4 * pen


def _valid(centers, angles, eps=1e-8):
    seps = _all_seps(centers, angles)
    return bool(np.all(seps >= -eps))


def _nelder(x0, outer_ang, min_sep, iters=150):
    x0 = np.array(x0, dtype=float)
    res = minimize(_objective_with_pen, x0, args=(outer_ang, min_sep),
                   method='Nelder-Mead',
                   options={'maxiter': iters, 'xatol': 1e-6, 'fatol': 1e-8,
                            'adaptive': True})
    return res.x


def _refine(x, outer_ang, min_sep):
    """Alternating: fix outer angle -> Nelder on inner; then fix inner -> optimize outer."""
    x = np.array(x, dtype=float)
    for _ in range(4):
        if time.time() > DEADLINE:
            break
        x = _nelder(x, outer_ang, min_sep, iters=120)
        centers = x[: 2 * N].reshape(N, 2)
        angles = x[2 * N:]
        def outer_obj(oa):
            return _outer_side(centers, angles, oa)
        res = minimize(outer_obj, np.array([outer_ang]), method='Nelder-Mead',
                       options={'maxiter': 60, 'xatol': 1e-5, 'fatol': 1e-9})
        outer_ang = float(res.x[0])
    return x, outer_ang


def _seed_configs():
    configs = []
    d = SQ3  # center distance for touching hexagons (flat-to-flat)
    # 1) hexagonal cluster: center + 6 + 4 of second ring
    pts = [(0.0, 0.0)]
    for k in range(6):
        a = k * math.pi / 3
        pts.append((d * math.cos(a), d * math.sin(a)))
    for k in range(6):
        a = k * math.pi / 3
        pts.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    for k in range(6):
        a = k * math.pi / 3 + math.pi / 6
        pts.append((2 * d * math.cos(a), 2 * d * math.sin(a)))
    pts = pts[:N]
    configs.append((np.array(pts, dtype=float).reshape(-1, 2), np.zeros(N)))
    # 2) linear
    xs = np.arange(N) - (N - 1) / 2.0
    configs.append((np.column_stack([xs * d, np.zeros(N)]), np.zeros(N)))
    # 3) two staggered rows
    pts = []
    for i in range(6):
        pts.append(((i - 2.5) * d, 0.0))
    for i in range(5):
        pts.append(((i - 2.0) * d, 1.5 * d))
    configs.append((np.array(pts, dtype=float).reshape(-1, 2), np.zeros(N)))
    # 4) compact 3-4-3-1 grid-ish
    pts = []
    row_y = [0.0, 1.5 * d, 3.0 * d, 4.5 * d]
    row_x = [(-1.0 * d, 0.0, 1.0 * d),
             (-0.5 * d, 0.5 * d, 1.5 * d, -1.5 * d),
             (-1.0 * d, 0.0, 1.0 * d),
             (0.0,)]
    for y, xs in zip(row_y, row_x):
        for xx in xs:
            pts.append((xx, y))
    pts = pts[:N]
    configs.append((np.array(pts, dtype=float).reshape(-1, 2), np.zeros(N)))
    # 5) random
    rng = np.random.default_rng(0)
    c = rng.uniform(-2 * d, 2 * d, (N, 2))
    configs.append((c, rng.uniform(-0.2, 0.2, N)))
    return configs


def construct():
    t0 = time.time()
    np.random.seed(7)

    best = None
    best_S = float('inf')

    def consider(centers, angles, outer_ang):
        nonlocal best, best_S
        if not _valid(centers, angles):
            return
        S = _outer_side(centers, angles, outer_ang)
        if S < best_S:
            best_S = S
            best = (centers.copy(), angles.copy(), outer_ang)

    # Phase 1: from each seed, do alternating Nelder refinement
    for c, a in _seed_configs():
        if time.time() > DEADLINE - 20:
            break
        try:
            x = np.concatenate([c.ravel(), a])
            # light initial polish
            x, oa = _refine(x, 0.0, 1e-6)
            centers = x[: 2 * N].reshape(N, 2)
            angles = x[2 * N:]
            consider(centers, angles, oa)
        except Exception:
            pass

    # Phase 2: random restarts + refinement, using current best as a hot start
    rng = np.random.default_rng(123)
    d = SQ3
    budget = DEADLINE - 8
    while time.time() < budget:
        if best is not None and rng.random() < 0.5:
            c0 = best[0] + rng.normal(0, 0.05, (N, 2))
            a0 = best[1] + rng.normal(0, 0.05, N)
            oa0 = best[2] + rng.normal(0, 0.05)
        else:
            c0 = rng.uniform(-2 * d, 2 * d, (N, 2))
            a0 = rng.uniform(-0.3, 0.3, N)
            oa0 = rng.uniform(-0.3, 0.3)
        try:
            x = np.concatenate([c0.ravel(), a0])
            x, oa = _refine(x, oa0, 1e-6)
            centers = x[: 2 * N].reshape(N, 2)
            angles = x[2 * N:]
            consider(centers, angles, oa)
        except Exception:
            pass

    # Phase 3: final tight polish on the best solution (very small step, more iters)
    if best is not None and time.time() < DEADLINE - 3:
        try:
            c, a, oa = best
            x = np.concatenate([c.ravel(), a])
            x, oa = _refine(x, oa, 1e-7)
            centers = x[: 2 * N].reshape(N, 2)
            angles = x[2 * N:]
            consider(centers, angles, oa)
        except Exception:
            pass

    # Fallback (shouldn't happen)
    if best is None:
        c = np.column_stack([np.arange(N) * SQ3 - (N - 1) * SQ3 / 2.0, np.zeros(N)])
        a = np.zeros(N)
        best = (c, a, 0.0)
        best_S = _outer_side(c, a, 0.0)

    centers, angles, outer_ang = best
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]),
                      float(math.degrees(angles[i]))])
    # Outer hexagon centered at origin; report its rotation.
    return inner, (0.0, 0.0), float(best_S), float(math.degrees(outer_ang))
