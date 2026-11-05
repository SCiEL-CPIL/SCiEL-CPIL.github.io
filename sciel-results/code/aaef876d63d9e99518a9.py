import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
APH = SQRT3 / 2.0          # apothem of a unit hexagon = 1/sqrt(3)
BASE = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
START = time.time()
DEADLINE = 145.0
MARGIN = 5.0

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _hex_vertices(cx, cy, angle_deg, s=1.0):
    angles = BASE + np.deg2rad(angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])


def _point_in_polygon(pt, poly):
    sign = 0
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True


def _seg_seg_dist(p1, p2, p3, p4):
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    if a < 1e-12:
        t = np.clip(f / e, 0, 1)
        return np.linalg.norm(p1 - (p3 + t * d2))
    c = np.dot(d1, r)
    if e < 1e-12:
        s = np.clip(c / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - p3)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom > 1e-12:
        s = np.clip((b * f - c * e) / denom, 0, 1)
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = np.clip(c / a, 0, 1)
        elif t > 1:
            t = 1
            s = np.clip((b + f) / a, 0, 1)
        return np.linalg.norm((p1 + s * d1) - (p3 + t * d2))
    t = np.clip(f / e, 0, 1)
    return np.linalg.norm(p1 - (p3 + t * d2))


def _poly_separation(p1, p2):
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    min_dist = float('inf')
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        a1, a2 = p1[i], p1[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = p2[j], p2[(j + 1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist


def _outer_hex_vertices(cx, cy, side, angle_deg):
    angles = BASE + np.deg2rad(angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])


# ---------------------------------------------------------------------------
# Exact, vectorized support-function objective
# ---------------------------------------------------------------------------
def _normals(angle_deg):
    na = BASE + np.deg2rad(angle_deg + 30)
    return np.column_stack([np.cos(na), np.sin(na)])


def _exact_side(params, angle_deg):
    n = params.shape[0]
    x = params[:, 0]
    y = params[:, 1]
    a = params[:, 2]
    N6 = _normals(angle_deg)                       # (6,2)
    # For each inner hexagon, max projection of its 6 vertices on each normal.
    # verts = [x,y] + R(a) * unit_hex ;  max over 6 vertices.
    unit = np.column_stack([np.cos(BASE), np.sin(BASE)])      # (6,2)
    best = np.empty((n, 6))
    for i in range(n):
        ca, sa = math.cos(math.radians(a[i])), math.sin(math.radians(a[i]))
        R = np.array([[ca, -sa], [sa, ca]])
        verts = unit @ R.T + np.array([x[i], y[i]])           # (6,2)
        proj = N6 @ verts.T                                   # (6,6)
        best[i] = proj.max(axis=1)
    return best.max() + APH


def _smooth_side(params, angle_deg, beta):
    """Log-sum-exp smooth upper bound on the exact side length."""
    n = params.shape[0]
    x = params[:, 0]
    y = params[:, 1]
    a = params[:, 2]
    N6 = _normals(angle_deg)
    unit = np.column_stack([np.cos(BASE), np.sin(BASE)])
    # required side for every (inner, vertex, normal) triple
    req = np.empty((n, 6, 6))
    for i in range(n):
        ca, sa = math.cos(math.radians(a[i])), math.sin(math.radians(a[i]))
        R = np.array([[ca, -sa], [sa, ca]])
        verts = unit @ R.T + np.array([x[i], y[i]])
        proj = N6 @ verts.T
        req[i] = proj / APH
    flat = req.ravel()
    m = flat.max()
    if m <= 0:
        return 0.0
    shifted = beta * (flat - m)
    return m + math.log(np.sum(np.exp(shifted))) / beta


def _overlap_penalty(params, weight=50.0, thr=0.005):
    n = params.shape[0]
    pen = 0.0
    for i in range(n):
        v1 = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
        for j in range(i + 1, n):
            v2 = _hex_vertices(params[j, 0], params[j, 1], params[j, 2])
            sep = _poly_separation(v1, v2)
            if sep < thr:
                t = (thr - sep) / thr
                if t > 50:
                    pen += weight * (thr - sep)
                else:
                    pen += weight * thr * math.log1p(math.exp(t))
    return pen


def _is_valid(params, side, angle_deg, min_sep=0.0):
    outer = _outer_hex_vertices(0.0, 0.0, side, angle_deg)
    for i in range(params.shape[0]):
        hv = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
        for v in hv:
            if not _point_in_polygon(v, outer):
                return False
    for i in range(params.shape[0]):
        v1 = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
        for j in range(i + 1, params.shape[0]):
            v2 = _hex_vertices(params[j, 0], params[j, 1], params[j, 2])
            if _poly_separation(v1, v2) < min_sep:
                return False
    return True


# ---------------------------------------------------------------------------
# Analytic optimal lattice packing (the 11-hex flower cluster)
# ---------------------------------------------------------------------------
def _lattice_packing():
    v1 = np.array([SQRT3, 0.0])
    v2 = np.array([SQRT3 / 2, 1.5])
    # center + 6 neighbours + 4 second-ring for a compact 11-cluster
    selected = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1),
                (-1, 1), (1, 1), (-1, -1), (2, 0), (0, -2)]
    centers = [i * v1 + j * v2 for (i, j) in selected]
    inner = np.array([[float(p[0]), float(p[1]), 30.0] for p in centers])
    # best outer angle by coarse + fine sweep
    best_side, best_angle = float('inf'), 0.0
    for ang in np.arange(0, 60, 0.5):
        s = _exact_side(inner, ang)
        if s < best_side:
            best_side, best_angle = s, ang
    for ang in np.arange(best_angle - 0.5, best_angle + 0.5, 0.02):
        s = _exact_side(inner, ang)
        if s < best_side:
            best_side, best_angle = s, ang
    return inner, best_side, best_angle


# ---------------------------------------------------------------------------
# Angle polish (exact objective)
# ---------------------------------------------------------------------------
def _polish_angle(inner, angle, limit):
    best_side = _exact_side(inner, angle)
    best_angle = angle
    for lo, hi, step in [(-3, 3, 0.2), (-0.2, 0.2, 0.01)]:
        for ang in np.arange(angle + lo, angle + hi, step):
            if time.time() > limit:
                break
            s = _exact_side(inner, ang)
            if s < best_side:
                best_side, best_angle = s, ang
    if time.time() < limit:
        try:
            r = minimize_scalar(lambda a: _exact_side(inner, a),
                                bounds=(best_angle - 0.05, best_angle + 0.05),
                                method='bounded', options={'xatol': 1e-11})
            if r.success and r.fun < best_side:
                best_side, best_angle = r.fun, r.x
        except Exception:
            pass
    return best_side, best_angle


# ---------------------------------------------------------------------------
# Optimization objective (smooth surrogate + overlap penalty)
# ---------------------------------------------------------------------------
def _objective(params, angle_deg, beta, pen_w):
    p = params.reshape(N, 3)
    return _smooth_side(p, angle_deg, beta) + pen_w * _overlap_penalty(p)


def _optimize_inner(inner0, angle, beta, pen_w, limit, maxiter):
    def obj(p):
        if time.time() > limit:
            return float('inf')
        return _objective(p, angle, beta, pen_w)
    try:
        r = minimize(obj, inner0.ravel(), method='Nelder-Mead',
                     options={'maxiter': maxiter, 'xatol': 1e-8,
                              'fatol': 1e-9, 'adaptive': True})
        cand = r.x.reshape(N, 3)
        s = _exact_side(cand, angle)
        if _is_valid(cand, s, angle, min_sep=1e-8):
            return cand, s
    except Exception:
        pass
    return None, float('inf')


def construct():
    inner, best_side, best_angle = _lattice_packing()

    # Phase 1: global-ish multi-start with differential evolution
    if time.time() < DEADLINE - MARGIN - 40:
        try:
            span = 4.0
            bounds = [(-span, span)] * (2 * N) + [(-90, 90)] * N
            limit1 = DEADLINE - MARGIN - 40
            res = differential_evolution(
                lambda p: _objective(p, best_angle, beta=30.0, pen_w=50.0),
                bounds, seed=1, maxiter=60, popsize=12,
                tol=1e-4, mutation=(0.3, 1.0), recombination=0.7,
                polish=False,
                callback=lambda p, c: (p < limit1 or _stop()),
                init='sobol')
            if res.fun < best_side + 1.0:
                cand = res.x.reshape(N, 3)
                s = _exact_side(cand, best_angle)
                if _is_valid(cand, s, best_angle, min_sep=1e-8) and s < best_side:
                    inner, best_side = cand, s
        except Exception:
            pass

    # Phase 2: local refinement from best (adaptive beta, increasing smoothness)
    for beta, pen_w in [(30.0, 50.0), (100.0, 100.0), (300.0, 200.0)]:
        if time.time() > DEADLINE - MARGIN - 15:
            break
        cand, s = _optimize_inner(inner, best_angle, beta, pen_w,
                                  DEADLINE - MARGIN - 15, 700)
        if cand is not None and s < best_side:
            inner, best_side = cand, s

    # Phase 3: high-precision angle polish
    if time.time() < DEADLINE - MARGIN:
        best_side, best_angle = _polish_angle(inner, best_angle,
                                              DEADLINE - MARGIN)

    # Phase 4: final joint polish (inner + angle) at high smoothness
    if time.time() < DEADLINE - MARGIN - 8:
        cand, s = _optimize_inner(inner, best_angle, 500.0, 300.0,
                                  DEADLINE - MARGIN - 8, 400)
        if cand is not None and s < best_side:
            inner, best_side = cand, s
        if time.time() < DEADLINE - MARGIN:
            best_side, best_angle = _polish_angle(inner, best_angle,
                                                  DEADLINE - MARGIN)

    # Final validation + safety
    final_side = _exact_side(inner, best_angle)
    if not _is_valid(inner, final_side, best_angle, min_sep=1e-8):
        # fall back to the analytic lattice packing (always valid)
        inner, final_side, best_angle = _lattice_packing()
    best_side = final_side

    inner_rows = [[float(r[0]), float(r[1]), float(r[2])] for r in inner]
    return inner_rows, (0.0, 0.0), float(best_side), float(best_angle)


def _stop():
    return time.time() > DEADLINE - MARGIN
