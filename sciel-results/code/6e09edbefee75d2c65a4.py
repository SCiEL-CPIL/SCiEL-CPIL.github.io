import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 147.0
SQRT3 = math.sqrt(3.0)

# Optimal 11-hexagon cluster on triangular lattice
_OPTIMAL_POSITIONS = np.array([
    [0.0, 0.0],
    [2.0, 0.0], [-1.0, SQRT3], [1.0, SQRT3],
    [-1.0, -SQRT3], [1.0, -SQRT3],
    [3.0, 0.0], [-2.0, SQRT3], [2.0, SQRT3],
    [-2.0, -SQRT3], [2.0, -SQRT3]
], dtype=float)


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
    return apothem * 2.0 / SQRT3


def _outer_side_smooth(theta, centers, theta_outer, beta=50.0):
    """Soft-max version of outer side length for smoother optimization."""
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    proj_flat = proj.flatten()
    m = proj_flat.max()
    # Use log-sum-exp with numerical stability
    shifted = (proj_flat - m) * beta
    max_shifted = shifted.max()
    soft_max = m + (max_shifted + math.log(np.exp(shifted - max_shifted).sum())) / beta
    return soft_max * 2.0 / SQRT3


def _pairwise_penalty(centers, hard_margin=2.0):
    """Penalty for pairs of centers closer than hard_margin (overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, hard_margin - dpair)
    return np.sum(viol * viol)


def _objective(params, theta_outer):
    """Objective: outer side length + large penalty for overlap."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _objective_eps(params, theta_outer, eps=1e-4):
    """Objective with slightly relaxed pairwise constraint for smoother optimization."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers, hard_margin=2.0 - eps)
    return s + 1000.0 * pen


def _objective_smooth(params, theta_outer, beta=50.0):
    """Smooth objective using soft-max for outer side."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side_smooth(theta, centers, theta_outer, beta=beta)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer, tol=1e-6):
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
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """
    Use LP to find the optimal outer hexagon center and inradius
    for given inner hexagons and outer angle.
    Returns (cx, cy, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle_deg in inner_hexes:
        iangle = math.radians(iangle_deg)
        for k in range(6):
            a = iangle + math.pi * k / 3
            all_vertices.append((ix + math.cos(a), iy + math.sin(a)))

    n_pts = len(all_vertices)
    c = [0.0, 0.0, 1.0]
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
            side = t / (SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None


def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Find best outer hexagon angle and center via LP.
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

    if best_angle >= 0:
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

        if time.time() < start + time_limit:
            uf_lo = max(0.0, best_angle - 0.05)
            uf_hi = min(30.0, best_angle + 0.05)
            ultra_fine = np.arange(uf_lo, uf_hi + 0.001, 0.001)
            for angle_deg in ultra_fine:
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
    mode = seed % 8

    if mode < 4:
        if mode == 0:
            centers = _OPTIMAL_POSITIONS.copy()
        elif mode == 1:
            centers = _OPTIMAL_POSITIONS.copy() + rng.uniform(-0.02, 0.02, size=(N, 2))
        elif mode == 2:
            centers = _OPTIMAL_POSITIONS.copy() + rng.uniform(-0.05, 0.05, size=(N, 2))
        else:
            centers = _OPTIMAL_POSITIONS.copy() + rng.uniform(-0.1, 0.1, size=(N, 2))
    elif mode == 4:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 5:
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 6:
        u = (SQRT3, 0.0)
        v = (SQRT3 / 2, 1.5)
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                candidates.append((x * x + y * y, x, y))
        candidates.sort(key=lambda t: t[0])
        centers = np.array([(c[1], c[2]) for c in candidates[:N]])
    else:
        centers = _OPTIMAL_POSITIONS.copy() + rng.uniform(-0.2, 0.2, size=(N, 2))

    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter=400):
    """Single L-BFGS-B optimization run using smooth objective."""
    try:
        res = minimize(_objective_smooth, init_params, args=(theta_outer, 50.0),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11})
        return res
    except Exception:
        return None


def _polish(params, theta_outer):
    """High-precision polish with smooth objective then exact."""
    cur_best = _objective(params, theta_outer)
    best_params = params.copy()
    
    # Stage 1: Smooth objectives with increasing beta
    for beta in [30.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        try:
            res = minimize(_objective_smooth, best_params, args=(theta_outer, beta),
                           method='L-BFGS-B',
                           options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-12})
            val = _objective(res.x, theta_outer)
            if val < cur_best:
                cur_best = val
                best_params = res.x
        except Exception:
            pass
    
    # Stage 2: SLSQP for local refinement
    try:
        res = minimize(_objective, best_params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 1500, 'ftol': 1e-16, 'maxls': 100})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Stage 3: High-precision L-BFGS-B
    try:
        res = minimize(_objective, best_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-15})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Stage 4: Another SLSQP pass from the refined point
    try:
        res = minimize(_objective, best_params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-17, 'maxls': 100})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    return best_params


def _lp_refine(centers, theta, time_limit=6.0):
    """
    Use LP to find the best outer hexagon (center, angle) for given inner hexagons.
    Returns (cx, cy, side, angle_deg) or None.
    """
    inner_hexes = [[float(centers[i, 0]), float(centers[i, 1]),
                    float(math.degrees(theta))] for i in range(N)]
    return _find_best_outer_lp(inner_hexes, time_limit=time_limit)


def _joint_optimize(centers_init, theta_inner_init, theta_outer_init, maxiter=300):
    """
    Jointly optimize inner positions, inner rotation, and outer rotation.
    This is a structural change: instead of fixing outer angle and optimizing inner,
    we optimize all variables simultaneously.
    """
    def obj(params):
        c = params[:N * 2].reshape(N, 2)
        ti = params[N * 2]
        to = params[N * 2 + 1]
        s = _outer_side(ti, c, to)
        pen = _pairwise_penalty(c)
        return s + 1000.0 * pen
    
    def obj_smooth(params, beta=50.0):
        c = params[:N * 2].reshape(N, 2)
        ti = params[N * 2]
        to = params[N * 2 + 1]
        s = _outer_side_smooth(ti, c, to, beta=beta)
        pen = _pairwise_penalty(c)
        return s + 1000.0 * pen
    
    params = np.concatenate([centers_init.reshape(-1), [theta_inner_init, theta_outer_init]])
    
    best_params = params.copy()
    best_val = obj(params)
    
    # Multi-stage optimization with increasing smoothness
    for beta in [20.0, 30.0, 50.0, 100.0, 200.0, 500.0]:
        try:
            res = minimize(obj_smooth, best_params, args=(beta,),
                           method='L-BFGS-B',
                           options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11})
            val = obj(res.x)
            if val < best_val:
                best_val = val
                best_params = res.x
        except Exception:
            pass
    
    # Final polish with exact objective
    try:
        res = minimize(obj, best_params,
                       method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # SLSQP refinement
    try:
        res = minimize(obj, best_params,
                       method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-16, 'maxls': 100})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    c = best_params[:N * 2].reshape(N, 2)
    ti = best_params[N * 2]
    to = best_params[N * 2 + 1]
    return c, ti, to, best_val


def _lp_inner_refine(centers, theta, theta_outer, time_limit=5.0):
    """
    Use LP to refine inner hexagon positions for fixed outer hexagon.
    We minimize the maximum projection of all vertices onto each outer normal,
    subject to pairwise distance constraints (relaxed via soft constraints).
    
    This is a linear program: minimize t subject to
    n_k · (center_i + v_j) <= t for all k, i, j
    and ||center_i - center_j|| >= 2 (handled via penalty or relaxation)
    
    For the LP part, we fix the pairwise constraints and just minimize t.
    Returns refined centers or None.
    """
    start = time.time()
    
    # Compute all vertex offsets for the inner hexagons
    verts_local = _hex_vertices_local(theta)  # (6, 2)
    normals = _outer_side_normals(theta_outer)  # (6, 2)
    
    # Build LP: minimize t
    # Variables: 2*N center coords + 1 t = 2*N + 1
    n_vars = 2 * N + 1
    c = np.zeros(n_vars)
    c[-1] = 1.0  # minimize t
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        nx, ny = normals[k]
        for i in range(N):
            for j in range(6):
                vx, vy = verts_local[j]
                # n_k · (center_i + v_j) <= t
                # nx*(cx_i + vx) + ny*(cy_i + vy) <= t
                # nx*cx_i + ny*cy_i - t <= -(nx*vx + ny*vy)
                row = np.zeros(n_vars)
                row[2*i] = nx
                row[2*i+1] = ny
                row[-1] = -1.0
                A_ub.append(row)
                b_ub.append(-(nx * vx + ny * vy))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None)] * (2 * N) + [(0, None)],
                      method='highs')
        if res.success:
            new_centers = res.x[:2 * N].reshape(N, 2)
            # Check if this is valid (pairwise distances)
            if _is_valid(new_centers, theta, theta_outer, tol=1e-4):
                return new_centers
            # If not valid, try to project back
            # Simple approach: return None and let the caller handle it
    except Exception:
        pass
    
    return None


def construct():
    best_score = float('inf')
    best_result = None
    
    # Phase 1: Joint optimization of inner positions, inner rotation, and outer rotation
    # This is the structural change: we no longer fix the outer angle
    joint_inits = [
        (_OPTIMAL_POSITIONS.copy(), 0.0, 0.0),
        (_OPTIMAL_POSITIONS.copy() + np.random.RandomState(1).uniform(-0.05, 0.05, size=(N, 2)), 
         np.random.RandomState(2).uniform(-math.pi/6, math.pi/6), 
         np.random.RandomState(3).uniform(0, math.pi/6)),
        (_OPTIMAL_POSITIONS.copy() + np.random.RandomState(4).uniform(-0.1, 0.1, size=(N, 2)), 
         np.random.RandomState(5).uniform(-math.pi/6, math.pi/6), 
         np.random.RandomState(6).uniform(0, math.pi/6)),
        (np.zeros((N, 2)), 0.0, math.pi/12),
        (_OPTIMAL_POSITIONS.copy() + np.random.RandomState(7).uniform(-0.15, 0.15, size=(N, 2)), 
         np.random.RandomState(8).uniform(-math.pi/6, math.pi/6), 
         np.random.RandomState(9).uniform(0, math.pi/6)),
    ]
    
    for ci, ti, to in joint_inits:
        if time.time() > DEADLINE - 35:
            break
        c, ti_opt, to_opt, val = _joint_optimize(ci, ti, to, maxiter=400)
        if _is_valid(c, ti_opt, to_opt):
            s = _outer_side(ti_opt, c, to_opt)
            if s < best_score:
                best_score = s
                best_result = (c.copy(), ti_opt, to_opt)
    
    # Phase 2: Fixed outer angle optimization (original approach as fallback)
    outer_rotations = [0.0, math.pi / 6.0, math.pi / 3.0, math.pi / 12.0, math.pi / 4.0]
    seed = 10
    
    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 30:
            init_params = _make_init(seed)
            seed += 1
            res = _optimize_one(init_params, theta_o, maxiter=400)
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
    
    # Phase 3: LP-based outer hexagon optimization
    if best_result is not None and time.time() < DEADLINE - 15:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=12.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)
    
    # Phase 4: Polish with multiple outer angle candidates
    if best_result is not None and time.time() < DEADLINE - 8:
        centers, theta, theta_o = best_result
        for theta_o_try in [theta_o, 0.0, math.pi / 6.0, math.pi / 12.0]:
            if time.time() > DEADLINE - 8:
                break
            params = np.concatenate([centers.reshape(-1), [theta]])
            polished = _polish(params, theta_o_try)
            centers_p = polished[:N * 2].reshape(N, 2)
            theta_p = polished[N * 2]
            if _is_valid(centers_p, theta_p, theta_o_try):
                s_p = _outer_side(theta_p, centers_p, theta_o_try)
                if s_p < best_score:
                    best_score = s_p
                    best_result = (centers_p.copy(), theta_p, theta_o_try)
    
    # Phase 5: LP inner refinement
    if best_result is not None and time.time() < DEADLINE - 3:
        centers, theta, theta_o = best_result
        refined = _lp_inner_refine(centers, theta, theta_o, time_limit=3.0)
        if refined is not None:
            s_r = _outer_side(theta, refined, theta_o)
            if s_r < best_score:
                best_score = s_r
                best_result = (refined.copy(), theta, theta_o)
    
    # Phase 6: Final LP outer refinement
    if best_result is not None and time.time() < DEADLINE - 1:
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
    
    if best_result is None:
        centers = _OPTIMAL_POSITIONS.copy()
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0
    
    centers, theta, theta_o = best_result
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
