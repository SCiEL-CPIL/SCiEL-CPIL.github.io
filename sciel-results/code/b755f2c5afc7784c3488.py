import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
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

def _soft_min_value(areas, beta):
    """Stable log-sum-exp based soft minimum."""
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    return -log_sum / beta

def _normalize_points(pts):
    """Center and normalize points to max radius 1."""
    center = np.mean(pts, axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 1e-10:
        pts = pts / max_norm
    return pts

def _hull_points_from_params(params, n_hull):
    """Generate hull points from parameters (angles and radii)."""
    angles = params[:n_hull]
    radii = params[n_hull:2*n_hull]
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def _interior_points_from_params(params, n_interior, hull_pts):
    """Generate interior points from parameters."""
    start = 2 * 7  # After 7 hull angles and 7 hull radii
    angles = params[start:start + n_interior]
    radii = params[start + n_interior:start + 2 * n_interior]
    return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))

def _build_points_from_params(params):
    """Build full 14-point configuration from parameter vector."""
    n_hull = 7
    n_interior = 7
    hull_pts = _hull_points_from_params(params, n_hull)
    interior_pts = _interior_points_from_params(params, n_interior, hull_pts)
    return np.vstack([hull_pts, interior_pts])

def _objective_hierarchical(flat_params):
    """Hierarchical objective: optimize hull + interior jointly."""
    params = flat_params
    if np.any(~np.isfinite(params)):
        return 1.0
    pts = _build_points_from_params(params)
    if np.max(np.linalg.norm(pts, axis=1)) < 1e-6:
        return 1.0
    pts = _normalize_points(pts)
    s = _score(pts)
    return -s if s > 0 else 1.0

def _objective_hierarchical_soft(flat_params, beta=100.0):
    """Soft hierarchical objective."""
    params = flat_params
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _build_points_from_params(params)
    if np.max(np.linalg.norm(pts, axis=1)) < 1e-6:
        return 1e10
    pts = _normalize_points(pts)
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

def _generate_hierarchical_inits():
    """Generate diverse initial parameters for hierarchical optimization."""
    inits = []
    n_hull = 7
    n_interior = 7
    
    # 1. Regular 7-gon hull + regular 7-gon interior (rotated)
    hull_angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
    hull_radii = np.ones(n_hull)
    interior_angles = np.linspace(0, 2 * np.pi, n_interior, endpoint=False) + np.pi / n_hull
    interior_radii = 0.6 * np.ones(n_interior)
    params = np.concatenate([hull_angles, hull_radii, interior_angles, interior_radii])
    inits.append(params)
    
    # 2. Perturbed versions
    for seed in range(8):
        rng = np.random.default_rng(seed)
        hull_angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False) + rng.normal(0, 0.1, n_hull)
        hull_radii = 1.0 + rng.normal(0, 0.05, n_hull)
        hull_radii = np.maximum(hull_radii, 0.8)
        interior_angles = np.linspace(0, 2 * np.pi, n_interior, endpoint=False) + rng.uniform(0, 2*np.pi, 1)
        interior_radii = 0.5 + rng.uniform(0, 0.3, n_interior)
        params = np.concatenate([hull_angles, hull_radii, interior_angles, interior_radii])
        inits.append(params)
    
    # 3. Different interior radius patterns
    for r_base in [0.4, 0.5, 0.6, 0.7]:
        hull_angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
        hull_radii = np.ones(n_hull)
        interior_angles = np.linspace(0, 2 * np.pi, n_interior, endpoint=False) + np.pi / n_hull
        interior_radii = r_base * np.ones(n_interior)
        params = np.concatenate([hull_angles, hull_radii, interior_angles, interior_radii])
        inits.append(params)
    
    # 4. Mixed: some interior points closer to hull
    for seed in range(5):
        rng = np.random.default_rng(seed + 100)
        hull_angles = np.linspace(0, 2 * np.pi, n_hull, endpoint=False)
        hull_radii = np.ones(n_hull)
        interior_angles = np.sort(rng.uniform(0, 2 * np.pi, n_interior))
        interior_radii = 0.3 + rng.uniform(0, 0.5, n_interior)
        params = np.concatenate([hull_angles, hull_radii, interior_angles, interior_radii])
        inits.append(params)
    
    return inits

def _worst_triangle_anchor_refinement(pts, max_iter=50, step_size=0.05):
    """Iteratively push the vertices of the worst triangle outward."""
    pts = pts.copy()
    for _ in range(max_iter):
        areas = _all_areas(pts)
        min_idx = np.argmin(areas)
        i, j, k = IDX_I[min_idx], IDX_J[min_idx], IDX_K[min_idx]
        
        # Compute gradient direction: push each vertex away from the centroid of the triangle
        centroid = (pts[i] + pts[j] + pts[k]) / 3.0
        
        for idx in [i, j, k]:
            direction = pts[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
                # Check if pushing outward maintains convexity (simple heuristic)
                new_pt = pts[idx] + step_size * direction
                # Ensure point stays within reasonable bounds
                if np.linalg.norm(new_pt) < 2.0:
                    pts[idx] = new_pt
        
        # Renormalize
        pts = _normalize_points(pts)
        
        # Check if improvement
        new_min_area = _compute_min_triangle_area(pts)
        if new_min_area <= areas[min_idx] * (1 - 1e-6):
            break
    
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

    # === PHASE 1: Hierarchical Differential Evolution ===
    # Parameters: 7 hull angles, 7 hull radii, 7 interior angles, 7 interior radii = 28 params
    bounds = []
    for _ in range(7):
        bounds.append((0, 2 * np.pi))  # hull angles
    for _ in range(7):
        bounds.append((0.8, 1.2))  # hull radii
    for _ in range(7):
        bounds.append((0, 2 * np.pi))  # interior angles
    for _ in range(7):
        bounds.append((0.2, 0.9))  # interior radii
    
    if remaining_time() > 70.0:
        try:
            res_de = differential_evolution(
                _objective_hierarchical,
                bounds,
                seed=42,
                maxiter=150,
                popsize=25,
                tol=1e-5,
                mutation=(0.4, 1.0),
                recombination=0.9,
                polish=False,
                init='latinhypercube',
                updating='immediate',
                workers=1
            )
            if res_de.fun < 0:
                pts14 = _build_points_from_params(res_de.x)
                pts14 = _normalize_points(pts14)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    if remaining_time() > 50.0:
        try:
            res_de2 = differential_evolution(
                _objective_hierarchical,
                bounds,
                seed=123,
                maxiter=100,
                popsize=20,
                tol=1e-5,
                mutation=(0.3, 0.8),
                recombination=0.85,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1
            )
            if res_de2.fun < 0:
                pts14 = _build_points_from_params(res_de2.x)
                pts14 = _normalize_points(pts14)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    # === PHASE 2: Local Refinement with Hierarchical Structure ===
    inits = _generate_hierarchical_inits()
    
    # Add best from DE if available
    if best_points is not None:
        # Try to extract parameters from best_points (approximate)
        # For now, just add the points directly to a general refinement
        pass
        
    # Add random inits
    rng = np.random.default_rng(999)
    for _ in range(5):
        hull_angles = np.sort(rng.uniform(0, 2 * np.pi, 7))
        hull_radii = 0.9 + rng.uniform(0, 0.1, 7)
        interior_angles = np.sort(rng.uniform(0, 2 * np.pi, 7))
        interior_radii = 0.3 + rng.uniform(0, 0.5, 7)
        params = np.concatenate([hull_angles, hull_radii, interior_angles, interior_radii])
        inits.append(params)

    for i, init_params in enumerate(inits):
        if remaining_time() < 15.0:
            break
        
        # Stage A: L-BFGS-B with soft hierarchical objective
        if remaining_time() > 12.0:
            try:
                res_lbfgs = minimize(
                    lambda x: _objective_hierarchical_soft(x, beta=100.0),
                    init_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if res_lbfgs.fun < 1e10:
                    pts14 = _build_points_from_params(res_lbfgs.x)
                    pts14 = _normalize_points(pts14)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
            except Exception:
                pass

        # Stage B: Nelder-Mead on hard hierarchical objective
        if remaining_time() > 8.0:
            try:
                res_nm = minimize(
                    _objective_hierarchical,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res_nm.fun < 0:
                    pts14 = _build_points_from_params(res_nm.x)
                    pts14 = _normalize_points(pts14)
                    cand_score = _score(pts14)
                    update_best(pts14, cand_score)
            except Exception:
                pass

        # Stage C: Worst-triangle anchor refinement
        if remaining_time() > 6.0 and best_points is not None:
            try:
                refined = _worst_triangle_anchor_refinement(best_points, max_iter=30, step_size=0.03)
                refined = _normalize_points(refined)
                cand_score = _score(refined)
                update_best(refined, cand_score)
            except Exception:
                pass

    # === PHASE 3: General refinement on best found ===
    if best_points is not None and remaining_time() > 10.0:
        try:
            init_flat = best_points.flatten()
            res_final = minimize(
                _objective_general if hasattr(_objective_general, '__call__') else _objective_hierarchical,
                init_flat,
                method='Powell',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if hasattr(res_final, 'fun') and res_final.fun < 0:
                pts14 = res_final.x.reshape(N, 2)
                pts14 = _normalize_points(pts14)
                cand_score = _score(pts14)
                update_best(pts14, cand_score)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
