import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 142.0  # safety margin before the 150 s kill
SQRT3 = math.sqrt(3.0)
SQRT3_HALF = SQRT3 / 2.0


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon."""
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side(theta, centers, theta_outer):
    """Minimal outer side length containing all inner hexagon vertices."""
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem = proj.max()
    return apothem / SQRT3_HALF


def _pairwise_penalty(centers):
    """Penalty for pairs of centers closer than 2 (overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return float(np.sum(viol * viol))


def _objective(params, theta_outer):
    """Objective: outer side length + large penalty for overlap."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer, tol=1e-5):
    """Check non-overlap and containment."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    s = _outer_side(theta, centers, theta_outer)
    apothem = s * SQRT3_HALF
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """
    LP: optimal outer-hexagon center + inradius for given inner hexagons and
    outer angle.  Returns (cx, cy, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle_deg in inner_hexes:
        iangle = math.radians(iangle_deg)
        for k in range(6):
            a = iangle + math.pi * k / 3
            all_vertices.append((ix + math.cos(a), iy + math.sin(a)))

    c = [0.0, 0.0, 1.0]  # minimize t (inradius)
    A_ub = []
    b_ub = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        for vx, vy in all_vertices:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / SQRT3_HALF
            return cx, cy, side
    except Exception:
        pass
    return None


def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Search the outer-hexagon angle (0..30 deg, hexagonal symmetry) with LP.
    Coarse grid of 0.5 deg, then a fine 0.01 deg grid around the best.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    coarse = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    if best_side < float('inf'):
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.01)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle


def _make_init(seed):
    """Generate initial parameters for optimization."""
    rng = np.random.RandomState(seed)
    mode = seed % 5
    sqrt3 = math.sqrt(3.0)

    if mode == 0:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float)
    elif mode == 3:
        u = (sqrt3, 0.0)
        v = (sqrt3 / 2, 1.5)
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                candidates.append((x * x + y * y, x, y))
        candidates.sort(key=lambda t: t[0])
        centers = np.array([(c[1], c[2]) for c in candidates[:N]])
    else:
        positions = [(0, 0), (2, 0), (-1, sqrt3), (1, sqrt3), (-1, -sqrt3), (1, -sqrt3),
                     (3, 0), (-2, sqrt3), (2, sqrt3), (-2, -sqrt3), (2, -sqrt3)]
        centers = np.array(positions, dtype=float) + rng.uniform(-0.1, 0.1, size=(N, 2))

    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    return np.concatenate([centers.reshape(-1), [theta]])


def _optimize_one(init_params, theta_outer, maxiter=350):
    """Single L-BFGS-B optimization run (fixed outer angle)."""
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11})
        return res
    except Exception:
        return None


def _smooth_objective(params):
    """
    Epigraph / smooth surrogate: simultaneously optimizes inner positions,
    inner angle AND outer angle.  The outer side is a smooth soft-max of the
    12 half-plane projections (log-sum-exp), so the outer angle is a free
    continuous variable.  Plus a quadratic overlap penalty.
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    theta_outer = params[N * 2 + 1]

    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T  # (66, 6)
    proj = np.concatenate([proj, -proj], axis=1)  # (66, 12)

    # log-sum-exp for numerical stability
    m = proj.max(axis=0)
    lse = m + np.log(np.sum(np.exp(proj - m), axis=0))
    beta = 80.0
    apothem = float(np.mean(lse) / beta)  # smooth max of the 12 projections
    s = apothem / SQRT3_HALF

    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _smooth_polish(params, theta_outer, maxiter=600):
    """High-precision polish that also refines the outer angle."""
    p0 = np.concatenate([params, [theta_outer]])
    try:
        res = minimize(_smooth_objective, p0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.fun < _smooth_objective(p0):
            return res.x
    except Exception:
        pass
    return params


def _polish(params, theta_outer):
    """High-precision polish (fixed outer angle)."""
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 300, 'ftol': 1e-15})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
        if res.success or res.fun < _objective(params, theta_outer):
            return res.x
    except Exception:
        pass
    return params


def _lp_refine(centers, theta, time_limit=6.0):
    """LP outer-hexagon search for fixed inner hexagons."""
    inner_hexes = [[float(centers[i, 0]), float(centers[i, 1]),
                    float(math.degrees(theta))] for i in range(N)]
    return _find_best_outer_lp(inner_hexes, time_limit=time_limit)


def construct():
    best_score = float('inf')
    best_result = None  # (centers, theta, theta_outer)

    # ---- Phase 1: multi-start continuous optimization (fixed outer angle) ----
    outer_rotations = [0.0, math.pi / 6.0, math.pi / 3.0]
    seed = 0
    maxiter = 350

    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 35:  # reserve time for polish + LP
            init_params = _make_init(seed)
            seed += 1
            res = _optimize_one(init_params, theta_o, maxiter)
            if res is None:
                continue
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if not _is_valid(centers, theta, theta_o):
                continue
            s = _outer_side(theta, centers, theta_o)
            if s < best_score:
                best_score = s
                best_result = (centers.copy(), theta, theta_o)

    if best_result is None:
        # Fallback: a valid hexagonal lattice (may be large but safe)
        sqrt3 = math.sqrt(3.0)
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
        theta = 0.0
        theta_o = 0.0
        s = _outer_side(theta, centers, theta_o)
        best_score = s
        best_result = (centers, theta, theta_o)

    # ---- Phase 2: smooth epigraph polish (also refines outer angle) ----
    if time.time() < DEADLINE - 20:
        centers, theta, theta_o = best_result
        params = np.concatenate([centers.reshape(-1), [theta]])
        polished = _smooth_polish(params, theta_o)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]
        theta_o_p = polished[N * 2 + 1]
        if _is_valid(centers_p, theta_p, theta_o_p):
            s_p = _outer_side(theta_p, centers_p, theta_o_p)
            if s_p < best_score:
                best_score = s_p
                best_result = (centers_p.copy(), theta_p, theta_o_p)

    # ---- Phase 3: fixed-angle high-precision polish ----
    if time.time() < DEADLINE - 12:
        centers, theta, theta_o = best_result
        params = np.concatenate([centers.reshape(-1), [theta]])
        polished = _polish(params, theta_o)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]
        if _is_valid(centers_p, theta_p, theta_o):
            s_p = _outer_side(theta_p, centers_p, theta_o)
            if s_p < best_score:
                best_score = s_p
                best_result = (centers_p.copy(), theta_p, theta_o)

    # ---- Phase 4: LP outer-hexagon refinement ----
    if time.time() < DEADLINE - 6:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=8.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)

    # ---- Phase 5: one final LP refinement on the polished solution ----
    if time.time() < DEADLINE - 1:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=4.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)

    centers, theta, theta_o = best_result
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
