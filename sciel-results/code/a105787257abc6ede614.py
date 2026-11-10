# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 500.0  # safety margin before 630s kill


def _rot(a_deg):
    a = np.deg2rad(a_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _hex_vertices(cx, cy, angle_deg, s=1.0):
    a = np.deg2rad(angle_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)


def _hex_normals(angle_deg):
    a = np.deg2rad(angle_deg)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.stack([np.cos(th), np.sin(th)], axis=1)


def _pair_gap(vA, vB, normals):
    projA = vA @ normals.T
    projB = vB @ normals.T
    gap = np.maximum(projA.min(0) - projB.max(0), projB.min(0) - projA.max(0))
    return gap.max()


def _build(base, f, phi_deg, theta_deg, c):
    R = _rot(phi_deg)
    centers = (base * f) @ R.T
    verts = []
    for k in range(centers.shape[0]):
        verts.append(_hex_vertices(centers[k, 0], centers[k, 1], phi_deg))
    allv = np.vstack(verts)
    a = np.deg2rad(theta_deg)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.stack([np.cos(th), np.sin(th)], axis=1)
    vals = allv @ normals.T - (normals @ c)
    s = (2.0 / np.sqrt(3.0)) * vals.max()
    # min separation gap over all pairs (same orientation -> 6 distinct normals)
    hn = _hex_normals(phi_deg)
    mingap = np.inf
    for i in range(centers.shape[0]):
        for j in range(i + 1, centers.shape[0]):
            mingap = min(mingap, _pair_gap(verts[i], verts[j], hn))
    return s, allv, centers, mingap


def _objective(params, base, f):
    phi, theta, ox, oy = params
    s, *_ = _build(base, f, phi, theta, np.array([ox, oy]))
    return s


def _coarse_rank(base):
    best = (np.inf, 0.0, 0.0)
    for phi in np.arange(0, 60, 10):
        for theta in np.arange(0, 60, 10):
            s, *_ = _build(base, 1.0, phi, theta, np.zeros(2))
            if s < best[0]:
                best = (s, phi, theta)
    return best


def _refine(base, f, phi0, theta0):
    x0 = np.array([phi0, theta0, 0.0, 0.0])
    res = minimize(_objective, x0, args=(base, f), method="Nelder-Mead",
                   options={"maxiter": 400, "xatol": 1e-5, "fatol": 1e-7})
    if res.success or res.fun < np.inf:
        return res.x
    # fallback: local grid around x0
    best = (np.inf, x0)
    for dphi in np.arange(-12, 12.1, 3):
        for dth in np.arange(-12, 12.1, 3):
            for ox in np.arange(-1.5, 1.51, 0.5):
                for oy in np.arange(-1.5, 1.51, 0.5):
                    p = np.array([phi0 + dphi, theta0 + dth, ox, oy])
                    v = _objective(p, base, f)
                    if v < best[0]:
                        best = (v, p)
    return best[1]


def _finalize(base, phi, theta, c, f0):
    f = f0
    for _ in range(200):
        s, allv, centers, mingap = _build(base, f, phi, theta, c)
        if mingap > 1e-6:
            break
        f += 1e-4
    # one more quick refinement at the chosen f
    p = _refine(base, f, phi, theta)
    phi, theta, ox, oy = p
    s, allv, centers, mingap = _build(base, f, phi, theta, np.array([ox, oy]))
    guard = 0
    while mingap <= 1e-6 and guard < 200:
        f += 1e-4
        s, allv, centers, mingap = _build(base, f, phi, theta, np.array([ox, oy]))
        guard += 1
    inner = np.column_stack([centers, np.full(centers.shape[0], phi)])
    outer = np.array([ox, oy, theta])
    return inner, outer, float(s), float(mingap)


def _fallback():
    inner = np.array([
        [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
        [-1.25, 2.17, 0], [1.25, 2.17, 0],
        [-1.25, -2.17, 0], [1.25, -2.17, 0],
        [-3.75, 2.17, 0], [3.75, 2.17, 0],
        [-3.75, -2.17, 0], [3.75, -2.17, 0],
    ])
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, 8.0


def hexagon_packing_11():
    # axial -> cartesian basis for pointy-top unit hexagons (vertex at 0 deg)
    a1 = np.array([1.5, np.sqrt(3) / 2.0])
    a2 = np.array([0.0, np.sqrt(3)])

    def axial(q, r):
        return np.array([q * a1[0] + r * a2[0], q * a1[1] + r * a2[1]])

    flower = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
             (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]

    from itertools import combinations
    candidates = []
    for combo in combinations(range(len(ring2)), 4):
        coords = flower + [ring2[k] for k in combo]
        base = np.array([axial(q, r) for (q, r) in coords])
        candidates.append(base)

    # coarse rank all shapes
    ranked = []
    for base in candidates:
        s, phi, theta = _coarse_rank(base)
        ranked.append((s, base, phi, theta))
    ranked.sort(key=lambda t: t[0])

    best = None
    best_s = np.inf
    for s0, base, phi0, theta0 in ranked[:25]:
        if time.time() > _DEADLINE:
            break
        p = _refine(base, 1.0, phi0, theta0)
        phi, theta, ox, oy = p
        inner, outer, s, mingap = _finalize(base, phi, theta, np.array([ox, oy]), 1.0006)
        if np.isfinite(s) and mingap > 1e-6 and s < best_s:
            best_s = s
            best = (inner, outer, s)

    if best is not None:
        return best
    return _fallback()


# EVOLVE-BLOCK-END
