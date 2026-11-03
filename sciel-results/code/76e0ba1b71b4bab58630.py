import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7
DEADLINE = 140.0  # Safety margin before 150s limit

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()
N_TRIS = len(IDX_I)

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _soft_min_value(areas, beta):
    """Stable log-sum-exp based soft minimum."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta

def _sym_params_to_points(params):
    """Generate 14 points from 7 angles and 7 radii (centrally symmetric)."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _objective_sym_soft(params, beta=200.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    points = np.vstack([half, -half])
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    sm = _soft_min_value(areas, beta)
    if sm < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / sm

def _objective_sym_soft_multi(params, betas=None):
    """Multi-beta soft objective for smoother landscape."""
    if betas is None:
        betas = [50.0, 150.0, 400.0]
    vals = []
    for b in betas:
        vals.append(_objective_sym_soft(params, beta=b))
    return np.mean(vals)

def _objective_sym_hard(params):
    """Hard objective: -score for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    points = np.vstack([half, -half])
    if np.any(~np.isfinite(points)):
        return 1e10
    s = _score(points)
    return -s

def _generate_sym_inits():
    """Generate diverse initial parameter vectors for centrally symmetric optimization."""
    inits = []
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    
    # Regular heptagon
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated regular
    for rot in [0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 1.7, 2.1, 2.5, 3.0]:
        inits.append(np.concatenate([base_angles + rot, np.ones(7)]))
    
    # Non-uniform angles with near-unity radii
    for seed in range(30):
        rng = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.2, 7)
        radii = np.maximum(radii, 0.4)
        inits.append(np.concatenate([angles, radii]))
    
    # Alternating radius patterns
    patterns = [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.0, 0.7, 0.9, 0.7, 1.0, 0.7, 0.9],
        [1.2, 0.9, 1.1, 0.9, 1.2, 0.9, 1.1],
        [1.0, 0.6, 0.9, 0.6, 1.0, 0.6, 0.9],
        [0.9, 1.1, 0.8, 1.1, 0.9, 1.1, 0.8],
        [1.0, 0.85, 1.0, 0.85, 1.0, 0.85, 1.0],
        [1.05, 0.95, 1.05, 0.95, 1.05, 0.95, 1.05],
        [1.1, 0.8, 1.0, 0.85, 1.05, 0.9, 0.95],
    ]
    for p in patterns:
        inits.append(np.concatenate([base_angles + 0.15, np.array(p)]))
    
    # Jittered regular
    for js in [0.03, 0.06, 0.1, 0.15, 0.2, 0.3]:
        rng = np.random.default_rng(999 + int(js * 100))
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js * 0.4, 7)
        r = np.maximum(r, 0.4)
        inits.append(np.concatenate([a, r]))
    
    # Two-ring: 3 inner + 4 outer
    for r_in in [0.3, 0.4, 0.5, 0.6, 0.7]:
        a_in = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        angles = np.concatenate([a_in, a_out])
        radii = np.concatenate([np.full(3, r_in), np.ones(4)])
        inits.append(np.concatenate([angles, radii]))
    
    # Three-ring: 2+3+2
    for r_in, r_mid in [(0.35, 0.65), (0.45, 0.75), (0.55, 0.85)]:
        a_in = np.linspace(0, 2 * np.pi, 2, endpoint=False)
        a_mid = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 6
        a_out = np.linspace(0, 2 * np.pi, 2, endpoint=False) + np.pi / 2
        angles = np.concatenate([a_in, a_mid, a_out])
        radii = np.concatenate([np.full(2, r_in), np.full(3, r_mid), np.ones(2)])
        inits.append(np.concatenate([angles, radii]))
    
    return inits

def _normalize_points(pts):
    """Center and normalize points."""
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm
    return pts

def construct():
    start_time = time.time()
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def remaining_time():
        return DEADLINE - (time.time() - start_time)

    # === PHASE 1: Differential Evolution in symmetric space ===
    def de_objective(flat):
        if np.any(~np.isfinite(flat)):
            return 1.0
        angles = flat[:7]
        radii = flat[7:]
        radii = np.maximum(radii, 0.1)
        half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = np.vstack([half, -half])
        if np.allclose(pts, pts[0]):
            return 1.0
        s = _score(pts)
        return -s if s > 0 else 1.0

    bounds = [(-np.pi, np.pi)] * 7 + [(0.1, 2.5)] * 7

    if remaining_time() > 45.0:
        try:
            res_de = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-6,
                mutation=(0.4, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                workers=1
            )
            if res_de.fun < 0:
                pts14 = _sym_params_to_points(res_de.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    if remaining_time() > 30.0:
        try:
            res_de2 = differential_evolution(
                de_objective,
                bounds,
                seed=123,
                maxiter=80,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 0.8),
                recombination=0.85,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if res_de2.fun < 0:
                pts14 = _sym_params_to_points(res_de2.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    # === PHASE 2: Local refinement of DE results and diverse inits ===
    sym_inits = _generate_sym_inits()
    
    if best_points is not None:
        center = np.mean(best_points, axis=0)
        pts_c = best_points - center
        half_pts = pts_c[:7]
        radii_est = np.linalg.norm(half_pts, axis=1)
        radii_est = np.maximum(radii_est, 0.1)
        angles_est = np.arctan2(half_pts[:, 1], half_pts[:, 0])
        order = np.argsort(angles_est)
        sym_flat = np.concatenate([angles_est[order], radii_est[order]])
        sym_inits.insert(0, sym_flat)

    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    sym_inits.append(np.concatenate([base_angles, np.ones(7)]))

    for i, flat_init in enumerate(sym_inits):
        if remaining_time() < 25.0:
            break
        
        # Stage A: L-BFGS-B with multi-beta soft objective
        if remaining_time() > 20.0:
            try:
                res_lbfgs = minimize(
                    _objective_sym_soft_multi,
                    flat_init,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                if res_lbfgs.fun < 1e10:
                    pts14 = _sym_params_to_points(res_lbfgs.x)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
                    flat_init = res_lbfgs.x
            except Exception:
                pass

        # Stage B: Nelder-Mead on hard objective
        if remaining_time() > 15.0:
            try:
                res_nm = minimize(
                    _objective_sym_hard,
                    flat_init,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.fun < 0:
                    pts14 = _sym_params_to_points(res_nm.x)
                    cand_score = -res_nm.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

        # Stage C: Powell on hard objective
        if remaining_time() > 10.0:
            try:
                res_pow = minimize(
                    _objective_sym_hard,
                    flat_init,
                    method='Powell',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                if res_pow.fun < 0:
                    pts14 = _sym_params_to_points(res_pow.x)
                    cand_score = -res_pow.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # === PHASE 3: General (non-symmetric) local search from best ===
    if best_points is not None and remaining_time() > 20.0:
        def _objective_general(flat_points):
            pts = flat_points.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            return -_score(pts)

        def _objective_general_soft(flat_points, beta=200.0):
            pts = flat_points.reshape(N, 2)
            if np.any(~np.isfinite(pts)):
                return 1e10
            areas = _all_areas(pts)
            if len(areas) == 0 or np.min(areas) < 1e-12:
                return 1e10
            sm = _soft_min_value(areas, beta)
            if sm < 1e-12:
                return 1e10
            hull_area = _compute_hull_area(pts)
            if hull_area < 1e-10:
                return 1e10
            return hull_area / sm

        pts_norm = _normalize_points(best_points)
        flat_gen = pts_norm.flatten()

        if remaining_time() > 15.0:
            try:
                res = minimize(
                    lambda x: _objective_general_soft(x, beta=200.0),
                    flat_gen,
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
                flat_gen = res.x
            except Exception:
                pass

        if remaining_time() > 10.0:
            try:
                res = minimize(
                    _objective_general,
                    flat_gen,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < 0:
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass

    # === PHASE 4: Random restarts around best ===
    if best_points is not None and remaining_time() > 12.0:
        rng2 = np.random.default_rng(77)
        for k in range(10):
            if remaining_time() < 8.0:
                break
            noise_scale = 0.02 + 0.015 * k
            noise = rng2.normal(0, noise_scale, size=(N, 2))
            perturbed = _normalize_points(best_points + noise)
            try:
                res = minimize(
                    lambda x: _objective_general_soft(x, beta=300.0),
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                cand_pts = res.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
