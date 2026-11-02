"""Hexagon packing: 11 unit hexagons in a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    a = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        th = a + np.deg2rad(60 * k)
        verts.append((cx + side * np.cos(th), cy + side * np.sin(th)))
    return np.array(verts)

def _build_cluster():
    d = np.sqrt(3.0)
    rows = [1, 2, 3, 2, 1]
    positions = []
    y = 0.0
    for i, count in enumerate(rows):
        x_start = -(count - 1) * 0.5 * d
        for j in range(count):
            positions.append((x_start + j * d, y))
        y += 1.5
    arr = np.array(positions)
    arr -= arr.mean(axis=0)
    return arr

def _outer_hex_violation(verts_all, ocx, ocy, oside, oangle_deg):
    a = np.deg2rad(oangle_deg)
    inr = oside * np.sqrt(3.0) / 2.0
    normals = []
    for k in range(6):
        th = a + np.deg2rad(30 + 60 * k)
        normals.append((np.cos(th), np.sin(th)))
    normals = np.array(normals)
    rel = verts_all - np.array([ocx, ocy])
    d = rel @ normals.T
    viol = d.max(axis=1) - inr
    return viol.max()

def _smooth_max(viol, beta=40.0):
    v = viol
    m = v.max()
    if m < -50:
        return m
    return m + np.log(np.sum(np.exp((v - m) * beta)) / len(v)) / beta

def _overlap_penalty(centers, beta=40.0):
    n = len(centers)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            gap = np.sqrt(3.0) - d
            if gap > -1e-6:
                pen += np.exp(beta * gap)
    return pen

def _objective(params, cluster0, n):
    ocx, ocy = params[0], params[1]
    oside = params[2]
    oangle = params[3]
    centers = cluster0.copy()
    angles = np.zeros(n)
    idx = 4
    for i in range(n):
        centers[i] = centers[i] + np.array([params[idx], params[idx + 1]])
        idx += 2
    for i in range(n):
        angles[i] = params[idx]
        idx += 1

    verts_list = [_hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(n)]
    verts_all = np.vstack(verts_list)

    viol = _outer_hex_violation(verts_all, ocx, ocy, oside, oangle)
    sm = _smooth_max(viol, beta=40.0)
    op = _overlap_penalty(centers)
    return oside + 200.0 * max(sm, 0.0) + 50.0 * op

def _min_side(ocx, ocy, oangle, verts):
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _outer_hex_violation(verts, ocx, ocy, mid, oangle) <= 1e-9:
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    deadline = START_TIME + TIME_LIMIT
    cluster0 = _build_cluster()
    n = len(cluster0)
    assert n == N

    best_params = None
    best_val = float('inf')

    verts0 = np.vstack([_hex_vertices(c[0], c[1], 0.0) for c in cluster0])
    inr_needed = 0.0
    for k in range(6):
        th = np.deg2rad(30 + 60 * k)
        nrm = np.array([np.cos(th), np.sin(th)])
        inr_needed = max(inr_needed, (verts0 @ nrm).max())
    oside0 = inr_needed * 2.0 / np.sqrt(3.0) + 0.05

    init = np.zeros(4 + 2 * n + n)
    init[2] = oside0
    idx = 4
    for i in range(n):
        idx += 2
    for i in range(n):
        idx += 1

    def obj(p):
        return _objective(p, cluster0, n)

    try:
        res = minimize(obj, init, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass

    rng = np.random.default_rng(7)
    for trial in range(10):
        if time.time() > deadline - 15:
            break
        p = init.copy()
        p[3] = rng.uniform(-30, 30)
        for i in range(n):
            p[4 + 2 * i] = rng.uniform(-0.3, 0.3)
            p[4 + 2 * i + 1] = rng.uniform(-0.3, 0.3)
        try:
            r = minimize(obj, p, method='Nelder-Mead',
                         options={'maxiter': 250, 'xatol': 1e-7, 'fatol': 1e-8, 'adaptive': True})
            if r.fun < best_val:
                best_val = r.fun
                best_params = r.x
        except Exception:
            pass

    ocx, ocy = best_params[0], best_params[1]
    oangle = best_params[3]
    centers = cluster0.copy()
    angles = np.zeros(n)
    idx = 4
    for i in range(n):
        centers[i] = centers[i] + np.array([best_params[idx], best_params[idx + 1]])
        idx += 2
    for i in range(n):
        angles[i] = best_params[idx]
        idx += 1

    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(n)]
    verts_all = np.vstack([_hex_vertices(c[0], c[1], c[2]) for c in inner])
    oside = _min_side(ocx, ocy, oangle, verts_all)

    return inner, (float(ocx), float(ocy)), float(oside), float(oangle)
