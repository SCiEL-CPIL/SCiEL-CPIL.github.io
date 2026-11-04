"""Hull-based packing of 11 unit hexagons in a regular hexagon (SCiEL evolves the EVOLVE-BLOCK)."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0  # apothem of a unit hexagon
# unit-hexagon vertex offset directions (relative to its own center)
_VOFF = np.array([(math.cos(math.radians(30.0 + 60.0 * k)),
                   math.sin(math.radians(30.0 + 60.0 * k))) for k in range(6)], dtype=float)
# 6 directions for the outer-hexagon support-function extents
_OUTDIR = np.array([(math.cos(math.radians(30.0 + 60.0 * k)),
                     math.sin(math.radians(30.0 + 60.0 * k))) for k in range(6)], dtype=float)
_OUTDIR_T = _OUTDIR.T  # shape (6,2)


def _hex_vertex_offsets(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _VOFF @ R.T  # (6,2) rotated offsets


def _hex_vertices_vec(centers, angles_rad):
    """centers (N,2), angles_rad (N,) -> (N,6,2) absolute vertex positions."""
    offs = np.stack([_hex_vertex_offsets(a) for a in angles_rad], axis=0)  # (N,6,2)
    return centers[:, None, :] + offs  # (N,6,2)


def _convex_hull(points):
    """Andrew monotone chain. points (M,2) -> hull (H,2) CCW."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    lower = []
    for p in pts:
        while len(lower) >= 2 and np.cross(lower[-1] - lower[-2], p - lower[-2]) <= 1e-15:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and np.cross(upper[-1] - upper[-2], p - upper[-2]) <= 1e-15:
            upper.pop()
        upper.append(p)
    return np.vstack([lower[:-1], upper[:-1]])


def _outer_side_from_hull(hull):
    """Minimal outer-hexagon side length enclosing the hull (exact, any orientation)."""
    if hull.shape[0] == 0:
        return 0.0
    c = hull.mean(axis=0)
    h = hull - c
    proj = h @ _OUTDIR_T  # (H,6)
    ext = np.abs(proj).max(axis=0)  # (6,)
    return float(ext.max() / APOTHEM)


def _poly_area_signed(v):
    n = v.shape[0]
    if n < 3:
        return 0.0
    x1, y1 = v[:, 0], v[:, 1]
    x2, y2 = v[(np.arange(n) + 1) % n, 0], v[(np.arange(n) + 1) % n, 1]
    return float(0.5 * np.sum(x1 * y2 - x2 * y1))


def _clip(subject, clip):
    out = list(subject)
    m = len(clip)
    for i in range(m):
        if not out:
            break
        inp = out
        out = []
        A, B = clip[i], clip[(i + 1) % m]
        edge = B - A

        def side(p):
            return np.dot(p - A, edge)

        def inter(P, Q):
            d1, d2 = side(P), side(Q)
            if abs(d1 - d2) < 1e-15:
                return P
            t = d1 / (d1 - d2)
            return P + t * (Q - P)

        S = inp[-1]
        for E in inp:
            sIn = side(S) >= -1e-12
            eIn = side(E) >= -1e-12
            if eIn:
                if not sIn:
                    out.append(inter(S, E))
                out.append(E)
            elif sIn:
                out.append(inter(S, E))
            S = E
    return out


def _overlap_area(v1, v2):
    clipped = _clip(v1, v2)
    if len(clipped) < 3:
        return 0.0
    return abs(_poly_area_signed(np.array(clipped)))


def _overlap_penalty(vertices, T=0.02):
    """Smooth penalty for pairwise overlap. vertices (N,6,2)."""
    Nn = vertices.shape[0]
    if Nn < 2:
        return 0.0
    a = np.zeros((Nn, Nn))
    for i in range(Nn):
        for j in range(i + 1, Nn):
            ov = _overlap_area(vertices[i], vertices[j])
            a[i, j] = ov
            a[j, i] = ov
    s = a.sum(axis=1)
    mx = s.max()
    if mx < 1e-13:
        return 0.0
    e = np.exp((s - mx) / T)
    return T * (mx + math.log(e.sum()))


def _hull_side(params):
    centers = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:]
    verts = _hex_vertices_vec(centers, angles).reshape(-1, 2)
    return _outer_side_from_hull(_convex_hull(verts))


def _objective(params, T=0.02):
    centers = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:]
    verts = _hex_vertices_vec(centers, angles)
    side = _hull_side(params)
    return side + 1000.0 * _overlap_penalty(verts, T)


def _is_valid(params):
    centers = params[:2 * N].reshape(N, 2)
    angles = params[2 * N:]
    verts = _hex_vertices_vec(centers, angles)
    for i in range(N):
        for j in range(i + 1, N):
            if _overlap_area(verts[i], verts[j]) > 1e-8:
                return False
    return _hull_side(params) < 1e6


def _analytic_seed():
    eps = 1e-9
    centers = [(0.0, 0.0)]
    for k in range(6):
        ang = math.radians(60.0 * k)
        centers.append(((SQRT3 + eps) * math.cos(ang), (SQRT3 + eps) * math.sin(ang)))
    for k in range(4):
        ang = math.radians(30.0 + 60.0 * k)
        centers.append(((3.0 + 2 * eps) * math.cos(ang), (3.0 + 2 * eps) * math.sin(ang)))
    return centers, [0.0] * N


def _params_from(centers, angles):
    p = np.zeros(3 * N)
    p[:2 * N] = np.array(centers, dtype=float).reshape(-1)
    p[2 * N:] = np.array(angles, dtype=float)
    return p


def _rotated_centers(p0, r):
    c, s = math.cos(r), math.sin(r)
    R = np.array([[c, -s], [s, c]])
    cc = p0[:2 * N].reshape(N, 2) @ R.T
    return cc.reshape(-1)


def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    best = None
    best_side = float('inf')

    def consider(params):
        nonlocal best, best_side
        if not _is_valid(params):
            return
        s = _hull_side(params)
        if s < best_side - 1e-12:
            best_side = s
            best = params.copy()

    # ---- Analytic seed (concentric ring, all hexagons axis-aligned) ----
    centers, angles = _analytic_seed()
    p0 = _params_from(centers, angles)
    consider(p0)

    # ---- Phase 1: global rotation of the whole cluster (low-dim, fast) ----
    if time.time() < deadline - 30:
        def rot_obj(r):
            r = float(r[0])
            return _hull_side(np.concatenate([_rotated_centers(p0, r), p0[2 * N:] + r]))
        try:
            rres = minimize(rot_obj, np.array([0.0]), method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            consider(np.concatenate([_rotated_centers(p0, float(rres.x[0])),
                                     p0[2 * N:] + float(rres.x[0])]))
        except Exception:
            pass

    # ---- Phase 2: full per-hexagon optimization from several starts ----
    starts = [p0]
    if best is not None:
        starts.append(best.copy())
    rng = np.random.default_rng(0)
    for _ in range(3):
        pp = p0.copy()
        pp[:2 * N] += rng.normal(0, 0.15, size=2 * N)
        pp[2 * N:] += rng.normal(0, 0.10, size=N)
        starts.append(pp)

    for pstart in starts:
        if time.time() > deadline - 20:
            break
        try:
            r = minimize(_objective, pstart, args=(0.05,), method='L-BFGS-B',
                         options={'maxiter': 400, 'ftol': 1e-9, 'gtol': 1e-7})
            consider(r.x)
            if time.time() < deadline - 15:
                r2 = minimize(_objective, r.x, args=(0.01,), method='L-BFGS-B',
                              options={'maxiter': 400, 'ftol': 1e-11, 'gtol': 1e-9})
                consider(r2.x)
        except Exception:
            pass

    # ---- Phase 3: high-precision polish of the best found ----
    if best is not None and time.time() < deadline - 5:
        try:
            rp = minimize(_objective, best, args=(0.002,), method='L-BFGS-B',
                          options={'maxiter': 1200, 'ftol': 1e-13, 'gtol': 1e-11})
            consider(rp.x)
        except Exception:
            pass

    # ---- Build return value from the best valid configuration ----
    if best is None:
        best = p0
    centers = best[:2 * N].reshape(N, 2)
    angles_rad = best[2 * N:]
    centers = centers - centers.mean(axis=0)  # outer hexagon at (0,0)
    verts = _hex_vertices_vec(centers, angles_rad).reshape(-1, 2)
    side = _outer_side_from_hull(_convex_hull(verts))
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(angles_rad[i]))]
             for i in range(N)]
    return inner, (0.0, 0.0), float(side), 0.0
