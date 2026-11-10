# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_EPS = 1e-6


def _hex_vertices(cx, cy, s, a_deg):
    ang = np.radians(a_deg)
    th = ang + np.arange(6) * np.pi / 3.0
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)


def _unit_normals():
    th = np.arange(6) * np.pi / 3.0
    return np.stack([np.cos(th), np.sin(th)], axis=1)


_NORMS = _unit_normals()


def _gap(v1, v2):
    # v1, v2: (6,2) vertex arrays of two unit hexagons (same orientation 0)
    ns = _NORMS
    s1 = v1 @ ns.T
    s2 = v2 @ ns.T
    g1 = s2.min(axis=1) - s1.max(axis=1)   # normal outward from hex1
    g2 = s1.min(axis=1) - s2.max(axis=1)   # normal outward from hex2
    return float(np.max(np.concatenate([g1, g2])))


def _build_inner(params):
    p = params.reshape(N, 3)
    cx = p[:, 0]
    cy = p[:, 1]
    ang = p[:, 2]
    verts = np.empty((N, 6, 2))
    for i in range(N):
        verts[i] = _hex_vertices(cx[i], cy[i], 1.0, ang[i])
    return verts


def _outer_vertices(side):
    th = np.arange(6) * np.pi / 3.0
    return np.stack([side * np.cos(th), side * np.sin(th)], axis=1)


def _separation_penalty(verts):
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            g = _gap(verts[i], verts[j])
            if g < _EPS:
                pen += (_EPS - g) ** 2
    return pen


def _containment_penalty(verts, side):
    ov = _outer_vertices(side)
    pen = 0.0
    for i in range(N):
        for k in range(6):
            p1 = ov[k]
            p2 = ov[(k + 1) % 6]
            edge = p2 - p1
            nrm = np.array([edge[1], -edge[0]])
            nrm = nrm / np.linalg.norm(nrm)
            if np.dot(nrm, p1) < 0:
                nrm = -nrm
            d = (verts[i] - p1) @ nrm
            if d > _EPS:
                pen += (d - _EPS) ** 2
    return pen


def _objective(params, side):
    verts = _build_inner(params)
    return 1e6 * (_separation_penalty(verts) + _containment_penalty(verts, side))


def _is_valid(params, side):
    verts = _build_inner(params)
    for i in range(N):
        for j in range(i + 1, N):
            if _gap(verts[i], verts[j]) < _EPS:
                return False
    ov = _outer_vertices(side)
    for i in range(N):
        for k in range(6):
            p1 = ov[k]
            p2 = ov[(k + 1) % 6]
            edge = p2 - p1
            nrm = np.array([edge[1], -edge[0]])
            nrm = nrm / np.linalg.norm(nrm)
            if np.dot(nrm, p1) < 0:
                nrm = -nrm
            d = (verts[i] - p1) @ nrm
            if d > _EPS:
                return False
    return True


def _needed_side(params):
    verts = _build_inner(params)
    allp = verts.reshape(-1, 2)
    return float(np.max(np.linalg.norm(allp, axis=1)))


def hexagon_packing_11():
    deadline = time.time() + 600.0
    dx = 2.0
    dy = np.sqrt(3.0)
    init = np.array([
        [0.0, 0.0],
        [dx, 0.0],
        [-dx, 0.0],
        [dx / 2, dy],
        [-dx / 2, dy],
        [dx / 2, -dy],
        [-dx / 2, -dy],
        [3 * dx / 2, dy],
        [-3 * dx / 2, dy],
        [3 * dx / 2, -dy],
        [-3 * dx / 2, -dy],
    ])
    best_params = np.hstack([init, np.zeros((N, 1))]).flatten().copy()
    best_side = _needed_side(best_params) + 0.05

    # Iteratively tighten: shrink side, optimize positions to re-satisfy constraints.
    for outer_iter in range(60):
        if time.time() > deadline - 90.0:
            break
        cur_side = _needed_side(best_params) + 0.0005
        res = minimize(
            _objective, best_params, args=(cur_side,),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-12,
                     'adaptive': True},
        )
        cand = res.x
        if _is_valid(cand, cur_side):
            best_params = cand
            best_side = cur_side
        else:
            # candidate invalid; stop tightening
            break

    # Ensure final validity (safety net)
    if not _is_valid(best_params, best_side):
        base = _needed_side(best_params)
        for trial in range(300):
            if time.time() > deadline - 30.0:
                break
            s_try = base + 0.02 * (trial + 1)
            if _is_valid(best_params, s_try):
                best_side = s_try
                break
        else:
            best_side = base + 10.0

    # Bisection for tightest valid side
    verts = _build_inner(best_params)
    allp = verts.reshape(-1, 2)
    lo = float(np.max(np.linalg.norm(allp, axis=1)))
    hi = best_side
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        mid = 0.5 * (lo + hi)
        if _is_valid(best_params, mid):
            hi = mid
        else:
            lo = mid
    best_side = hi

    inner = best_params.reshape(N, 3)
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(best_side)


# EVOLVE-BLOCK-END
