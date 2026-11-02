import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0

# Unit hexagon corner offsets (pointy-top, side 1)
CORNER_ANGS = np.array([math.pi / 2 + k * math.pi / 3.0 for k in range(6)])
CORNER_OFF = np.column_stack([np.cos(CORNER_ANGS), np.sin(CORNER_ANGS)])


def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _corners(cx, cy, ang):
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return cx + cy * 0 + CORNER_OFF @ R.T + np.array([cx, cy]) * 0 + np.array([0, 0]) + (CORNER_OFF @ R.T) * 0 + np.array([cx, cy])


def _all_corners(params):
    # params: [x1,y1,a1, ..., xN,yN,aN, cx,cy, beta]
    n = N
    centers = params[:3 * n].reshape(n, 3)
    cx, cy = params[3 * n], params[3 * n + 1]
    corners = np.empty((n, 6, 2))
    for i in range(n):
        x, y, a = centers[i]
        c, s = math.cos(a), math.sin(a)
        R = np.array([[c, -s], [s, c]])
        corners[i] = CORNER_OFF @ R.T + np.array([x, y])
    return corners, cx, cy


def _smooth_obj(params, t_rep, t_cont, w_rep, w_cont):
    corners, cx, cy = _all_corners(params)
    n = N
    # Container: log-sum-exp over 6 face normals of max corner projection
    beta = params[3 * n + 2]
    normals = np.array([[math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    centered = corners.reshape(-1, 2) - np.array([cx, cy])
    projs = centered @ normals.T  # (36, 6)
    maxp = np.max(projs, axis=0)
    minp = np.min(projs, axis=0)
    allp = np.concatenate([maxp, -minp])
    m = np.max(allp)
    if m > 1000:
        return 1e12
    D = t_cont * (m + math.log(np.sum(np.exp((allp - m) / t_cont))))
    S = (D + APOTHEM) / APOTHEM
    # Repulsion: log-sum-exp over all pairwise corner distances
    C = corners.reshape(-1, 2)
    diffs = C[:, None, :] - C[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    # Exclude same-hexagon pairs and self
    mask = np.ones((36, 36), dtype=bool)
    for i in range(n):
        mask[6 * i:6 * i + 6, 6 * i:6 * i + 6] = False
    dmat = np.where(mask, dists, 1e12)
    # We want large distances; penalize small ones: use -log-sum-exp of (d - r)
    val = dmat - SQRT3
    m2 = np.min(val)
    if m2 < -100:
        return 1e12
    rep = -t_rep * (m2 + math.log(np.sum(np.exp((val - m2) / t_rep))))
    return w_cont * S + w_rep * rep


def _exact_S(corners, cx, cy, beta):
    normals = np.array([[math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    centered = corners.reshape(-1, 2) - np.array([cx, cy])
    projs = centered @ normals.T
    D = max(np.max(np.max(projs, axis=0)), -np.min(np.min(projs, axis=0)))
    return (D + APOTHEM) / APOTHEM, D


def _separation_ok(corners):
    C = corners.reshape(-1, 2)
    n = N
    for i in range(n):
        for j in range(i + 1, n):
            di = C[6 * i:6 * i + 6]
            dj = C[6 * j:6 * j + 6]
            d = di[:, None, :] - dj[None, :, :]
            dist = np.linalg.norm(d, axis=2)
            if np.min(dist) < SQRT3 - 1e-6:
                return False
    return True


def _anneal(start_params):
    best = float('inf')
    best_p = start_params.copy()
    t_rep = 0.5
    t_cont = 0.5
    w_rep = 1.0
    w_cont = 1.0
    for stage in range(6):
        if _time_left() < 3:
            break
        t_rep = max(0.02, 0.5 * (0.4 ** stage))
        t_cont = max(0.02, 0.5 * (0.4 ** stage))
        w_rep = 1.0 + 2.0 * stage
        try:
            res = minimize(_smooth_obj, best_p, method='L-BFGS-B',
                           args=(t_rep, t_cont, w_rep, w_cont),
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < best:
                # evaluate exact
                corners, cx, cy = _all_corners(res.x)
                beta = res.x[3 * N + 2]
                S, D = _exact_S(corners, cx, cy, beta)
                if S < best and _separation_ok(corners):
                    best = S
                    best_p = res.x.copy()
                elif S < best:
                    # keep smooth progress even if tiny overlap; will fix later
                    best_p = res.x.copy()
        except Exception:
            pass
    # final tightening
    if _time_left() > 2:
        try:
            res = minimize(_smooth_obj, best_p, method='L-BFGS-B',
                           args=(0.005, 0.005, 10.0, 1.0),
                           options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-11})
            corners, cx, cy = _all_corners(res.x)
            beta = res.x[3 * N + 2]
            S, D = _exact_S(corners, cx, cy, beta)
            if S < best and _separation_ok(corners):
                best = S
                best_p = res.x.copy()
        except Exception:
            pass
    return best, best_p


def _start_params(axial, beta):
    # axial -> cartesian
    pts = []
    for q, r in axial:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    params = np.zeros(3 * N + 3)
    for i in range(N):
        params[3 * i] = pts[i, 0]
        params[3 * i + 1] = pts[i, 1]
        params[3 * i + 2] = 0.0
    params[3 * N] = cx
    params[3 * N + 1] = cy
    params[3 * N + 2] = beta
    return params


def construct():
    if _time_left() < 5:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    templates = []
    # compact flower
    templates.append([(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                      (2, 0), (-2, 0), (0, 2), (0, -2)])
    # 3x3-ish block
    templates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
                      (0, 2), (1, 2), (2, 2), (1, 3), (2, 3)])
    # elongated
    templates.append([(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1),
                      (3, 1), (1, 2), (2, 2), (3, 2)])
    # diamond
    templates.append([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
                      (2, 0), (-2, 0), (0, 2), (0, -2)])
    # staggered
    templates.append([(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1),
                      (0, 2), (1, 2), (2, 2), (3, 2)])
    # two-row
    templates.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                      (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)])
    # triangular
    templates.append([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
                      (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)])

    best_S = float('inf')
    best_p = None

    for tmpl in templates:
        if _time_left() < 10:
            break
        for beta in [0.0, math.pi / 12.0, math.pi / 6.0 * 0.33, math.pi / 6.0 * 0.66]:
            if _time_left() < 8:
                break
            p0 = _start_params(tmpl, beta)
            S, p = _anneal(p0)
            if S < best_S:
                best_S = S
                best_p = p
        # jittered starts
        if _time_left() < 12:
            break
        rng = np.random.default_rng(abs(hash(tmpl)) % (2**32))
        for _ in range(3):
            if _time_left() < 8:
                break
            p0 = _start_params(tmpl, 0.0)
            jitter = rng.normal(0, 0.3, size=3 * N + 3)
            jitter[3 * N + 2] = 0.0  # keep beta
            p0j = p0 + jitter
            S, p = _anneal(p0j)
            if S < best_S:
                best_S = S
                best_p = p

    if best_p is None:
        p0 = _start_params(templates[0], 0.0)
        best_S, best_p = _anneal(p0)

    # extract
    corners, cx, cy = _all_corners(best_p)
    beta = best_p[3 * N + 2]
    angle_deg = math.degrees(beta)
    S, D = _exact_S(corners, cx, cy, beta)
    # safety: if overlap, scale positions out slightly
    if not _separation_ok(corners):
        # nudge apart via one more tight repulsion pass
        try:
            res = minimize(_smooth_obj, best_p, method='L-BFGS-B',
                           args=(0.002, 0.005, 50.0, 1.0),
                           options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-12})
            corners, cx, cy = _all_corners(res.x)
            beta = res.x[3 * N + 2]
            angle_deg = math.degrees(beta)
            S, D = _exact_S(corners, cx, cy, beta)
        except Exception:
            pass

    inner = []
    for i in range(N):
        x, y, a = best_p[3 * i], best_p[3 * i + 1], best_p[3 * i + 2]
        inner.append([float(x), float(y), float(math.degrees(a))])
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(angle_deg)
    return inner, center, side, rotation
