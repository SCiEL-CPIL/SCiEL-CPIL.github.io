import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

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
    """Compute soft-min of areas using log-sum-exp."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _objective_soft(flat_points, beta=50.0):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    soft_min = _soft_min_value(areas, beta)
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

def _objective_soft_multi_beta(flat_points, betas=None):
    """Multi-scale soft-min: average over multiple beta values for robust landscape."""
    if betas is None:
        betas = [20.0, 50.0, 100.0, 200.0]
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

def _objective_sym_soft(params, beta=150.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    soft_min_val = _soft_min_value(areas, beta)
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _objective_sym_soft_multi(params, betas=None):
    """Multi-scale soft objective for symmetric params."""
    if betas is None:
        betas = [50.0, 100.0, 200.0, 400.0]
    vals = []
    for b in betas:
        vals.append(_objective_sym_soft(params, beta=b))
    return np.mean(vals)

def _generate_sym_inits():
    """Generate initial parameter vectors for centrally symmetric optimization."""
    inits = []
    
    # Regular heptagon-based (7 equally spaced angles)
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated versions
    for rot in [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]))
    
    # Non-uniform angle patterns with varying radii
    rng = np.random.default_rng(42)
    for seed in range(25):
        rng = np.random.default_rng(seed * 11 + 5)
        angles = np.sort(rng.uniform(0, 2*np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.15, 7)
        radii = np.maximum(radii, 0.5)
        inits.append(np.concatenate([angles, radii]))
    
    # Pattern: alternating radii
    for r_pattern in [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.0, 0.7, 0.9, 0.7, 1.0, 0.7, 0.9],
        [1.2, 0.9, 1.1, 0.9, 1.2, 0.9, 1.1],
    ]:
        a = base_angles + 0.2
        inits.append(np.concatenate([a, np.array(r_pattern)]))
    
    # Jittered regular
    for js in [0.05, 0.1, 0.2]:
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js*0.5, 7)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
    return inits

def _normalize_points(pts):
    """Center and normalize points to unit norm."""
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm
    return pts

def _sa_sym(params_init, time_budget=30.0, rng_seed=42):
    """Simulated annealing in symmetric parameter space with adaptive step size."""
    rng = np.random.default_rng(rng_seed)
    params = params_init.copy()
    
    # Initial evaluation
    pts = _sym_params_to_points(params)
    current_score = _score(pts)
    if current_score <= 0:
        current_score = 1e-10
    
    best_params = params.copy()
    best_score = current_score
    
    # Adaptive SA parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    step_size = 0.1
    base_step = 0.1
    n_iter = 50000
    start = time.time()
    
    for i in range(n_iter):
        if time.time() - start > time_budget:
            break
        
        # Adaptive step size
        if i % 1000 == 0 and i > 0:
            # Reduce step size over time
            step_size = base_step * (T / 0.1)
        
        # Generate perturbation
        delta = rng.normal(0, step_size, size=14)
        new_params = params + delta
        
        # Keep radii positive
        new_params[7:] = np.maximum(new_params[7:], 0.1)
        
        # Evaluate
        new_pts = _sym_params_to_points(new_params)
        new_score = _score(new_pts)
        if new_score <= 0:
            new_score = 1e-10
        
        # Acceptance criterion
        if new_score > current_score:
            params = new_params
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta_score = new_score - current_score
            if delta_score > -T * 5:  # Avoid overflow
                prob = np.exp(delta_score / T)
                if rng.random() < prob:
                    params = new_params
                    current_score = new_score
        
        # Cool down
        T = max(T_min, T * 0.9995)
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8):
        pts = points_init.copy()
        pts = _normalize_points(pts)
        
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # 1. Generate diverse initializations in symmetric space
    sym_inits = _generate_sym_inits()
    
    # 2. Run SA from multiple diverse starts
    sa_results = []
    for i, flat_init in enumerate(sym_inits[:15]):  # Limit to 15 best inits
        if time.time() > deadline - 40.0:
            break
        
        remaining_time = deadline - 40.0 - time.time()
        sa_time = min(remaining_time / max(1, 15 - i), 8.0)
        if sa_time < 1.0:
            sa_time = 1.0
        
        best_params, best_sc = _sa_sym(flat_init, time_budget=sa_time, rng_seed=i*13+7)
        if best_sc > 0:
            pts14 = _sym_params_to_points(best_params)
            sa_results.append((best_sc, best_params))
            update_best(pts14, best_sc)
    
    # 3. Polish top SA results with Nelder-Mead
    sa_results.sort(key=lambda x: -x[0])
    for i, (sc, params) in enumerate(sa_results[:5]):
        if time.time() > deadline - 15.0:
            break
        
        pts14 = _sym_params_to_points(params)
        cand_pts, cand_score = refine(pts14, 'Nelder-Mead', maxiter=1000)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
        
        # Also try Powell
        if time.time() < deadline - 10.0:
            cand_pts2, cand_score2 = refine(pts14, 'Powell', maxiter=500)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 4. Non-symmetric refinement from best points
    if best_points is not None and time.time() < deadline - 10.0:
        cand_pts, cand_score = refine(best_points, 'Nelder-Mead', maxiter=1500)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
        
        if time.time() < deadline - 5.0:
            cand_pts2, cand_score2 = refine(best_points, 'Powell', maxiter=800)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 5. Final random restarts if time allows
    if best_points is not None and time.time() < deadline - 8.0:
        rng2 = np.random.default_rng(99)
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            noise = rng2.normal(0, 0.05, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=500)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
