import numpy as np
from scipy.optimize import minimize
import time

N = 12
# 6-fold symmetry: outer hexagon vertices at angles 0,60,...,300
_VERTS = np.array([np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))])
                   for a in range(0, 360, 60)])
_EDGE_MID = (_VERTS + np.roll(_VERTS, -1, axis=0)) / 2.0
# outward normals (unit), pointing away from center
_NORM = _EDGE_MID / np.linalg.norm(_EDGE_MID, axis=1, keepdims=True)


def _half_widths(r):
    """For a hexagon of circumradius r (centered, flat orientation), the
    distance from center to each of the 6 edges."""
    return np.array([r * np.cos(np.deg2rad(a)) for a in range(30, 330, 60)])


def _erode(r, s, m=120):
    """Vertices of the erosion (outer hexagon of circumradius s) minus a
    unit disk (inner circumradius 1). Returns M x 2 array."""
    hw = _half_widths(s)
    # distance from origin to each edge (before translation)
    d_edge = hw - r
    # build a polygon in (theta, rho) then to cartesian
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    cs = np.cos(th)
    sn = np.sin(th)
    # distance from origin to edge i at angle th: hw_i - (cs*nx_i + sn*ny_i)*s
    rho = np.min(hw[None, :] - s * (cs[:, None] * _NORM[0][None, :] + sn[:, None] * _NORM[1][None, :]), axis=1)
    # feasibility: rho must be >= d_edge for all edges
    if np.min(rho) < -1e-9:
        return None
    # clip to feasible region: intersect half-planes
    # start with the disk boundary, then clip by each edge
    pts = np.column_stack([rho * cs, rho * sn])
    for i in range(6):
        nx, ny = _NORM[i]
        # keep points with nx*x+ny*y <= hw_i - r
        keep = nx * pts[:, 0] + ny * pts[:, 1] <= (hw[i] - r) + 1e-9
        if not np.all(keep):
            # add intersection points with boundary
            x0, y0 = pts[:, 0], pts[:, 1]
            val = nx * x0 + ny * y0 - (hw[i] - r)
            # find sign changes along the closed loop
            inter = []
            for j in range(len(pts)):
                j2 = (j + 1) % len(pts)
                v0, v1 = val[j], val[j2]
                if (v0 <= 0) != (v1 <= 0):
                    t = v0 / (v0 - v1)
                    inter.append((x0[j] + t * (x0[j2] - x0[j]),
                                  y0[j] + t * (y0[j2] - y0[j])))
            pts = np.array([[x0[j], y0[j]] for j in range(len(pts)) if val[j] <= 0] + inter)
            if len(pts) < 3:
                return None
    return pts


def _in_poly(px, py, poly):
    """Point-in-convex-polygon (CCW)."""
    x0, y0 = poly[0]
    for i in range(len(poly) - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        if (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0) < -1e-9:
            return False
    return True


def _in_eroded(x, y, s, cache):
    key = round(s, 6)
    if key not in cache:
        cache[key] = _erode(1.0, s)
    poly = cache[key]
    if poly is None:
        return False
    return _in_poly(x, y, poly)


def _objective_flat(v, s, cache):
    pts = v.reshape(-1, 2)
    for i in range(N):
        if not _in_eroded(pts[i, 0], pts[i, 1], s, cache):
            return 1.0
    return 0.0


def _fit_side(pts):
    """Smallest s (by bisection) such that all pts are in the eroded region."""
    cache = {}
    lo, hi = 2.0, 12.0
    # ensure hi feasible
    if _objective_flat(pts.ravel(), hi, cache) > 0.5:
        return hi
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if _objective_flat(pts.ravel(), mid, cache) <= 0.5:
            hi = mid
        else:
            lo = mid
    return hi


def _coarse_grid(side, n=9):
    """Return list of (points, side) candidates from a coarse grid."""
    cands = []
    # candidate centers spread over a disk of radius ~ side
    R = side * 0.9
    xs = np.linspace(-R, R, n)
    base = np.array([[x, y] for x in xs for y in xs])
    # sample a subset to keep it cheap
    if len(base) > 200:
        idx = np.linspace(0, len(base) - 1, 200).astype(int)
        base = base[idx]
    for b in base:
        pts = np.tile(b, (N, 1))
        s = _fit_side(pts)
        cands.append((pts.copy(), s))
    cands.sort(key=lambda t: t[1])
    return cands


def _basin_polish(pts, side):
    """Local optimization of the side given fixed layout via center shifts."""
    cache = {}
    best = (pts.copy(), side)
    cur = pts.copy()
    cur_s = side
    for it in range(60):
        # random perturbation
        step = 0.5 * max(0.1, cur_s * 0.05)
        trial = cur + step * np.random.randn(*cur.shape)
        ts = _fit_side(trial)
        if ts < cur_s - 1e-6:
            cur, cur_s = trial, ts
            best = (cur.copy(), cur_s)
        else:
            # small local nudge
            trial2 = cur + 0.1 * step * np.random.randn(*cur.shape)
            ts2 = _fit_side(trial2)
            if ts2 < cur_s - 1e-7:
                cur, cur_s = trial2, ts2
                best = (cur.copy(), cur_s)
    return best


def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin under 150s

    # Start from a reasonable initial side (hexagonal cluster packing)
    # Place 12 hexagons in a compact hexagonal arrangement.
    def hex_cluster(side):
        # ring of 6 around a center, plus extra 5
        pts = np.zeros((N, 2))
        # center
        pts[0] = [0, 0]
        ang = np.deg2rad([0, 60, 120, 180, 240, 300])
        rad = 2.0  # distance between adjacent centers for touching hexes
        for i in range(6):
            pts[1 + i] = [rad * np.cos(ang[i]), rad * np.sin(ang[i])]
        # 5 more on an outer ring
        rad2 = 3.5
        ang2 = np.deg2rad([30, 90, 150, 210, 270])
        for i in range(5):
            pts[7 + i] = [rad2 * np.cos(ang2[i]), rad2 * np.sin(ang2[i])]
        return pts

    start_pts = hex_cluster(4.0)
    start_side = _fit_side(start_pts)
    best_pts, best_side = start_pts.copy(), start_side

    # Coarse grid seeds
    seeds = _coarse_grid(best_side)
    # Take the best few seeds plus the cluster start
    candidates = [(start_pts, start_side)] + seeds[:8]

    rng = np.random.default_rng(1234)
    for (p0, s0) in candidates:
        if time.time() > DEADLINE:
            break
        # run basin hopping from this seed
        for restart in range(3):
            if time.time() > DEADLINE:
                break
            trial_pts = p0 + 0.3 * max(0.2, s0 * 0.1) * rng.standard_normal(p0.shape)
            trial_side = _fit_side(trial_pts)
            if trial_side < best_side:
                best_pts, best_side = trial_pts.copy(), trial_side
            # polish
            pp, ss = _basin_polish(trial_pts, trial_side)
            if ss < best_side:
                best_pts, best_side = pp, ss
            if time.time() > DEADLINE:
                break

    # Final high-precision polish on the best
    if time.time() < DEADLINE:
        pp, ss = _basin_polish(best_pts, best_side)
        if ss < best_side:
            best_pts, best_side = pp, ss

    # Round to a clean representation; keep rotation 0 for all
    inner = [[float(best_pts[i, 0]), float(best_pts[i, 1]), 0.0] for i in range(N)]
    return inner, (0.0, 0.0), float(best_side), 0.0
