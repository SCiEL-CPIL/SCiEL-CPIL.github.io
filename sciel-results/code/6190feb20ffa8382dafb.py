# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _separation_ok(v1, n1, v2, n2):
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    return False


def _hexagons_separated(inner_data):
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        verts.append(_hex_vertices(x, y, 1.0, a))
        norms.append(_hex_normals(a))
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _contains_in_hex(inner_data, outer_data, side):
    ox, oy, oa = outer_data
    outer_v = _hex_vertices(ox, oy, side, oa)
    tol = -1e-6
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    if not _hexagons_separated(inner_data):
        return False
    if not _contains_in_hex(inner_data, outer_data, side):
        return False
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
    return True


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(60 * np.arange(6))
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    return np.vstack([dirs, -dirs])


# ---- fast, exact geometry used by the optimizer -------------------------

def _outer_normals(g):
    """6 outward edge normals (3 unique directions) of an outer hexagon at angle g."""
    a = np.deg2rad(g)
    base = a + np.deg2rad(30 + 60 * np.arange(6))
    return np.stack([np.cos(base), np.sin(base)], axis=1)


def _inner_verts(centers, g):
    """(66,2) vertices of all 11 unit hexagons (orientation g, unit side)."""
    a = np.deg2rad(g)
    ang = a + np.deg2rad(60 * np.arange(6))
    off = np.stack([np.cos(ang), np.sin(ang)], axis=1)            # (6,2)
    return centers[:, None, :] + off[None, :, :]                   # (11,6,2)


def _exact_side(centers, g):
    """Minimum outer side (centered at origin, angle g) containing all inner hexagons."""
    n = _outer_normals(g)
    uv = _inner_verts(centers, g)
    proj = np.einsum('ijk,jk->ij', uv, n)                          # (66,6)
    return (2.0 / np.sqrt(3.0)) * float(np.max(np.abs(proj)))


def _overlap_penalty(centers, margin=1e-5):
    """Smooth penalty: 0 iff every pair is separated by > 1e-6 (parallel hexagons)."""
    d = centers[:, None, :] - centers[None, :, :]                  # (11,11,2)
    proj = np.abs(np.einsum('ijk,jk->ij', d, _outer_normals(0.0)))  # (11,11,6)
    m = proj.max(axis=2)                                           # (11,11)
    thr = np.sqrt(3.0) + margin
    viol = np.maximum(0.0, thr - m)
    return float((viol * viol).sum())


def _build_inner(centers, g):
    inner = np.empty((11, 3))
    inner[:, 0] = centers[:, 0]
    inner[:, 1] = centers[:, 1]
    inner[:, 2] = g
    return inner


def _objective(params, margin=1e-5):
    g = params[0]
    centers = params[1:].reshape(11, 2)
    if not np.all(np.isfinite(centers)):
        return 1e6
    return _exact_side(centers, g) + 1000.0 * _overlap_penalty(centers, margin)


def _initial_centers():
    r1 = np.sqrt(3.0) * 1.005            # 6 ring (touching the center)
    r2 = 2.0 * np.sqrt(3.0) * 1.01       # 4 spikes
    c = [[0.0, 0.0]]
    for k in range(6):
        a = np.deg2rad(60 * k)
        c.append([r1 * np.cos(a), r1 * np.sin(a)])
    for k in [0, 2, 3, 5]:
        a = np.deg2rad(60 * k + 30)
        c.append([r2 * np.cos(a), r2 * np.sin(a)])
    return np.array(c, dtype=float)


def _fallback():
    centers = _initial_centers()
    g = 0.0
    inner = _build_inner(centers, g)
    outer = np.array([0.0, 0.0, g])
    side = _exact_side(centers, g)
    for s in (side, side + 0.5, side + 1.0, 8.0, 9.0, 10.0):
        if _valid(inner, outer, s):
            return inner, outer, s
    return inner, outer, side


def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 615.0

    def remaining():
        return deadline - time.time()

    base = _initial_centers()
    best_val = float('inf')
    best_params = None

    def consider(params):
        nonlocal best_val, best_params
        v = _objective(params)
        if v < best_val:
            best_val = v
            best_params = np.array(params, dtype=float)

    try:
        from scipy.optimize import differential_evolution, minimize
    except Exception:
        return _fallback()

    # ---- Phase 1: global exploration (differential evolution) ------------
    if remaining() > 150:
        try:
            lb = np.concatenate(([-30.0], -10.0 * np.ones(22)))
            ub = np.concatenate(([30.0], 10.0 * np.ones(22)))
            x0 = np.concatenate(([0.0], base.ravel()))
            de = differential_evolution(
                _objective, bounds=list(zip(lb, ub)), x0=x0, seed=0,
                maxiter=120, popsize=15, tol=1e-6, mutation=(0.3, 1.0),
                recombination=0.8, polish=False, init='latinhypercube',
                update_interval=1,
            )
            if de.x is not None:
                consider(de.x)
        except Exception:
            pass

    # ---- Phase 2: smooth polish (Powell) ---------------------------------
    start = best_params if best_params is not None else np.concatenate(([0.0], base.ravel()))
    if remaining() > 120:
        try:
            res = minimize(_objective, start, method='Powell',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.x is not None:
                consider(res.x)
        except Exception:
            pass

    # ---- Phase 3: tight local pass (Nelder-Mead) -------------------------
    if remaining() > 60:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-11, 'fatol': 1e-11,
                                    'adaptive': True})
            if res.x is not None:
                consider(res.x)
        except Exception:
            pass

    if best_params is None:
        return _fallback()

    g = float(best_params[0])
    centers = best_params[1:].reshape(11, 2)
    inner = _build_inner(centers, g)
    outer = np.array([0.0, 0.0, g])
    side = _exact_side(centers, g)

    if not _valid(inner, outer, side):
        # nudge the side up until the full geometric check passes
        for s in (side + 1e-6, side + 1e-5, side + 1e-4, side + 1e-3,
                  side + 1e-2, side + 0.1, side + 0.5, 8.0, 9.0, 10.0):
            if _valid(inner, outer, s):
                return inner, outer, s
        return _fallback()

    return inner, outer, side


# EVOLVE-BLOCK-END
