import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7
DEADLINE = 138.0  # Safety margin before 150s limit

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

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _objective_soft(flat_points, beta=100.0):
    """Soft-min objective: differentiable approximation of hull_area/min_area."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_soft_multi_beta(flat_points, betas=None):
    """Multi-scale soft-min: average over multiple beta values for robust landscape."""
    if betas is None:
        betas = [30.0, 80.0, 200.0, 500.0]
    vals = []
    for b in betas:
        vals.append(_objective_soft(flat_points, beta=b))
    return np.mean(vals)

# --- Centrally symmetric parameterization ---
# For N=14, we use 7 independent vertex pairs (v_i, -v_i).
# Parameterize by 7 angles and 7 radii: v_i = r_i * (cos(theta_i), sin(theta_i))

def _sym_params_to_points(params):
    """Generate 14 points from 7 angles and 7 radii (centrally symmetric)."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _objective_sym(params):
    """Hard objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    return -_score(points)

def _objective_sym_soft(params, beta=200.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _objective_sym_soft_multi(params, betas=None):
    """Multi-scale soft objective for symmetric params."""
    if betas is None:
        betas = [80.0, 200.0, 500.0]
    vals = []
    for b in betas:
        vals.append(_objective_sym_soft(params, beta=b))
    return np.mean(vals)

def _objective_feasibility(flat_points, t_init=None):
    """Feasibility-based objective using auxiliary variable t.
    Minimizes hull_area/t + penalty for triangles below t*hull_area."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    
    min_a = np.min(areas)
    if min_a < 1e-12:
        return 1e10
    
    # Use soft penalty: encourage all areas to be at least min_a
    # This helps avoid degenerate configurations
    target = min_a
    violations = np.maximum(0.0, target - areas)
    penalty = 0.1 * np.sum(violations ** 2)
    
    return hull_area / min_a + penalty

def _generate_sym_inits():
    """Generate initial parameter vectors for centrally symmetric optimization."""
    inits = []
    
    # Regular heptagon-based (7 equally spaced angles)
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated versions
    for rot in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2, 1.5, 1.7, 2.0, 2.3]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]))
    
    # Non-uniform angle patterns with varying radii
    for seed in range(40):
        rng = np.random.default_rng(seed * 13 + 7)
        angles = np.sort(rng.uniform(0, 2 * np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.25, 7)
        radii = np.maximum(radii, 0.3)
        inits.append(np.concatenate([angles, radii]))
    
    # Pattern: alternating radii
    for r_pattern in [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.0, 0.7, 0.9, 0.7, 1.0, 0.7, 0.9],
        [1.2, 0.9, 1.1, 0.9, 1.2, 0.9, 1.1],
        [1.0, 0.6, 0.9, 0.6, 1.0, 0.6, 0.9],
        [0.9, 1.1, 0.8, 1.1, 0.9, 1.1, 0.8],
        [1.0, 0.85, 1.0, 0.85, 1.0, 0.85, 1.0],
        [1.05, 0.95, 1.05, 0.95, 1.05, 0.95, 1.05],
    ]:
        a = base_angles + 0.15
        inits.append(np.concatenate([a, np.array(r_pattern)]))
    
    # Jittered regular
    for js in [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2]:
        rng = np.random.default_rng(999 + int(js * 100))
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js * 0.5, 7)
        r = np.maximum(r, 0.3)
        inits.append(np.concatenate([a, r]))
    
    # Two-ring symmetric: 3 inner + 4 outer
    for r_in in [0.35, 0.45, 0.55, 0.65]:
        a_in = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
        angles = np.concatenate([a_in, a_out])
        radii = np.concatenate([np.full(3, r_in), np.ones(4)])
        inits.append(np.concatenate([angles, radii]))
    
    # Three-ring: 2+3+2
    for r_in, r_mid in [(0.4, 0.7), (0.5, 0.8), (0.6, 0.85)]:
        a_in = np.linspace(0, 2 * np.pi, 2, endpoint=False)
        a_mid = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 6
        a_out = np.linspace(0, 2 * np.pi, 2, endpoint=False) + np.pi / 2
        angles = np.concatenate([a_in, a_mid, a_out])
        radii = np.concatenate([np.full(2, r_in), np.full(3, r_mid), np.ones(2)])
        inits.append(np.concatenate([angles, radii]))
    
    # Support-function style: nearly uniform angles with slight perturbation
    for pert in [0.02, 0.05, 0.1]:
        rng = np.random.default_rng(555 + int(pert * 100))
        a = base_angles + rng.uniform(-pert, pert, 7)
        r = 1.0 + rng.normal(0, pert * 0.3, 7)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
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

    # 1. Symmetric optimization (PRIMARY STRATEGY)
    # Centrally symmetric: 7 angles + 7 radii = 14 parameters
    sym_inits = _generate_sym_inits()
    
    for i, flat_init in enumerate(sym_inits):
        if remaining_time() < 30.0:
            break
        
        # Nelder-Mead on symmetric parameters with soft objective
        try:
            res_sym = minimize(_objective_sym_soft, flat_init, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
            if res_sym.fun < 1e10:
                pts14 = _sym_params_to_points(res_sym.x)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass
        
        # Powell on symmetric parameters with hard objective
        if remaining_time() > 25.0:
            try:
                res_sym_pow = minimize(_objective_sym, flat_init, method='Powell',
                                       options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
                if res_sym_pow.fun < 0:
                    pts14 = _sym_params_to_points(res_sym_pow.x)
                    cand_score = -res_sym_pow.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # 2. Differential Evolution on symmetric space
    if remaining_time() > 35.0:
        def de_sym_objective(flat):
            if np.any(~np.isfinite(flat)):
                return 1.0
            pts = _sym_params_to_points(flat)
            if np.allclose(pts, pts[0]):
                return 1.0
            s = _score(pts)
            return -s if s > 0 else 1.0
            
        try:
            bounds = [(-np.pi, np.pi)] * 7 + [(0.1, 2.0)] * 7
            res_de = differential_evolution(
                de_sym_objective, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=12, 
                tol=1e-5, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.fun < 0:
                pts14 = _sym_params_to_points(res_de.x)
                cand_score = -res_de.fun
                update_best(pts14, cand_score)
        except Exception:
            pass
        
        # Second DE run with different seed if time allows
        if remaining_time() > 20.0:
            try:
                res_de2 = differential_evolution(
                    de_sym_objective, 
                    bounds, 
                    seed=123, 
                    maxiter=50, 
                    popsize=10, 
                    tol=1e-5, 
                    mutation=(0.3, 0.8), 
                    recombination=0.8,
                    polish=True,
                    init='sobol'
                )
                if res_de2.fun < 0:
                    pts14 = _sym_params_to_points(res_de2.x)
                    cand_score = -res_de2.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # 3. General (non-symmetric) optimization from diverse initializations
    if remaining_time() > 30.0:
        initializations = []
        
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        initializations.append(x_reg)
        
        # Two concentric rings (7+7)
        for r1 in [0.4, 0.5, 0.6, 0.7, 0.8]:
            a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
            pts = np.vstack([
                r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
                np.column_stack([np.cos(a2), np.sin(a2)])
            ])
            initializations.append(pts)
        
        # Three rings (5+4+5)
        for r1, r2 in [(0.35, 0.65), (0.45, 0.75), (0.55, 0.85)]:
            a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
            a2 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
            a3 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
            pts = np.vstack([
                r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
                r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
                np.column_stack([np.cos(a3), np.sin(a3)])
            ])
            initializations.append(pts)
        
        # Elliptical
        for rx, ry in [(1.0, 0.65), (1.0, 0.8), (1.15, 0.75)]:
            a = np.linspace(0, 2 * np.pi, N, endpoint=False)
            initializations.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))
        
        # Random
        rng = np.random.default_rng(456)
        for _ in range(3):
            r = np.sqrt(rng.uniform(0, 1, N))
            theta = rng.uniform(0, 2 * np.pi, N)
            x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
            initializations.append(x_rand)

        for i, init_pts in enumerate(initializations):
            if remaining_time() < 25.0:
                break
            
            pts_norm = _normalize_points(init_pts)
            
            # Nelder-Mead
            try:
                res = minimize(_objective, pts_norm.flatten(), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                if res.fun < 0:
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass
            
            # Powell
            if remaining_time() > 20.0:
                try:
                    res = minimize(_objective, pts_norm.flatten(), method='Powell',
                                   options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
                    if res.fun < 0:
                        cand_pts = res.x.reshape(N, 2)
                        cand_score = -res.fun
                        update_best(cand_pts, cand_score)
                except Exception:
                    pass

    # 4. Final polish on the global best - multi-stage
    if best_points is not None and remaining_time() > 15.0:
        # Stage 1: Multi-beta soft refinement
        if remaining_time() > 12.0:
            try:
                pts_norm = _normalize_points(best_points)
                res_multi = minimize(lambda x: _objective_soft_multi_beta(x), pts_norm.flatten(),
                                     method='L-BFGS-B',
                                     options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-11})
                cand_pts = res_multi.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
            except Exception:
                pass
        
        # Stage 2: Hard Nelder-Mead with tight tolerance
        if remaining_time() > 10.0:
            try:
                pts_norm = _normalize_points(best_points)
                res = minimize(_objective, pts_norm.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                if res.fun < 0:
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass
        
        # Stage 3: Powell with tight tolerance
        if remaining_time() > 8.0:
            try:
                pts_norm = _normalize_points(best_points)
                res = minimize(_objective, pts_norm.flatten(), method='Powell',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < 0:
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass
        
        # Stage 4: If best is symmetric, refine in symmetric space
        if remaining_time() > 6.0:
            try:
                center = np.mean(best_points, axis=0)
                pts_centered = best_points - center
                sym_params_est = []
                used = set()
                for i in range(N_HALF):
                    best_j = -1
                    best_dist = np.inf
                    for j in range(N_HALF, N):
                        if j in used:
                            continue
                        d = np.linalg.norm(pts_centered[i] + pts_centered[j])
                        if d < best_dist:
                            best_dist = d
                            best_j = j
                    if best_j >= 0:
                        used.add(best_j)
                        v = (pts_centered[i] - pts_centered[best_j]) / 2
                        angle = np.arctan2(v[1], v[0])
                        radius = np.linalg.norm(v)
                        sym_params_est.extend([angle, max(radius, 0.1)])
                if len(sym_params_est) == 14:
                    sym_flat = np.array(sym_params_est)
                    res_sym_final = minimize(_objective_sym, sym_flat, method='Nelder-Mead',
                                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
                    if res_sym_final.fun < 0:
                        pts14 = _sym_params_to_points(res_sym_final.x)
                        cand_score = -res_sym_final.fun
                        update_best(pts14, cand_score)
            except Exception:
                pass

    # 5. Random restarts if time allows
    if best_points is not None and remaining_time() > 12.0:
        rng2 = np.random.default_rng(77)
        for k in range(10):
            if remaining_time() < 8.0:
                break
            noise = rng2.normal(0, 0.03 + 0.01 * k, size=(N, 2))
            perturbed = _normalize_points(best_points + noise)
            try:
                res = minimize(_objective, perturbed.flatten(), method='Nelder-Mead',
                               options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                if res.fun < 0:
                    cand_pts = res.x.reshape(N, 2)
                    cand_score = -res.fun
                    update_best(cand_pts, cand_score)
            except Exception:
                pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
