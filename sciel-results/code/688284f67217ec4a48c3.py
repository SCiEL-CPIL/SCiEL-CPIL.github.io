import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 142.0  # Safety margin before 150s kill

def _params_to_points(params):
    """Convert angular/radial perturbation params to 2D points.
    params shape (N, 2): [angle_offset, radial_scale] for each vertex of regular N-gon.
    Ensures strict convexity by construction."""
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angle_offsets = params[:, 0]
    radial_scales = params[:, 1]
    # Constrain angles to maintain ordering (convexity)
    angles = base_angles + angle_offsets
    # Sort by angle to maintain convex polygon order
    order = np.argsort(angles)
    angles_sorted = angles[order]
    radials_sorted = radial_scales[order]
    # Ensure positive radii
    radials_sorted = np.maximum(radials_sorted, 0.1)
    x = radials_sorted * np.cos(angles_sorted)
    y = radials_sorted * np.sin(angles_sorted)
    return np.column_stack([x, y])

def _points_to_params(pts):
    """Convert 2D points to angular/radial params relative to centroid."""
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    radii = np.sqrt(centered[:, 0]**2 + centered[:, 1]**2)
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # Unwrap angle differences
    angle_diffs = angles - base_angles
    # Handle wrapping
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([angle_diffs, radii])

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas."""
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    i_idx = np.array([c[0] for c in idx])
    j_idx = np.array([c[1] for c in idx])
    k_idx = np.array([c[2] for c in idx])
    v1 = pts[j_idx] - pts[i_idx]
    v2 = pts[k_idx] - pts[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _soft_min(areas, tau):
    """Soft minimum via log-sum-exp."""
    m = np.min(areas)
    shifted = areas - m
    shifted = np.clip(shifted, 0, 50.0 / tau)
    sum_exp = np.sum(np.exp(-tau * shifted))
    return m - (1.0 / tau) * np.log(sum_exp)

def _min_triangle_area(pts):
    """Minimum triangle area."""
    return np.min(_triangle_areas(pts))

def _hull_area(pts):
    """Convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    ma = _min_triangle_area(pts)
    return ma / ha

def _normalize(pts):
    """Center and scale to unit hull area."""
    pts = pts - np.mean(pts, axis=0)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return pts
    return pts / np.sqrt(ha)

def _neg_objective_param(params):
    """Negative score after normalization, in param space."""
    pts = _params_to_points(params)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _smooth_objective_param(params, tau=80.0):
    """Smooth surrogate in param space."""
    pts = _params_to_points(params)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, tau)
    return -sm / ha

def _neg_objective(flat):
    """Negative score after normalization (raw Cartesian)."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _smooth_objective(flat, tau=50.0):
    """Smooth surrogate in raw Cartesian space."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, tau)
    return -sm / ha

def _local_refine(pts, max_iters=300, base_step=0.01, deadline=None):
    """Fast local refinement: perturb points in worst triangles using in-place updates."""
    pts = pts.copy()
    current = _score(pts)
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    idx_arr = np.array(idx)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        tol = 1e-6
        worst_mask = np.abs(areas - min_a) < tol
        worst_tris = np.where(worst_mask)[0][:30]
        worst_pts = set()
        for ti in worst_tris:
            for p in idx_arr[ti]:
                worst_pts.add(p)
        
        if not worst_pts:
            break
        
        improved = False
        step = base_step * (0.7 ** (it // 30))
        for pi in sorted(worst_pts):
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                          (-step, -step), (step, step), (-step, step), (step, -step)]:
                orig_x, orig_y = pts[pi, 0], pts[pi, 1]
                pts[pi, 0] += dx
                pts[pi, 1] += dy
                s = _score(pts)
                if s > current + 1e-12:
                    current = s
                    improved = True
                    break
                else:
                    pts[pi, 0] = orig_x
                    pts[pi, 1] = orig_y
            if improved:
                break
        if not improved:
            if step < 1e-6:
                break
            base_step *= 0.5
    return pts, current

def _local_refine_param(params, max_iters=300, base_step=0.01, deadline=None):
    """Local refinement in parameter space."""
    params = params.copy()
    pts = _params_to_points(params)
    pts = _normalize(pts)
    current = _score(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        tol = 1e-6
        worst_mask = np.abs(areas - min_a) < tol
        worst_tris = np.where(worst_mask)[0][:30]
        
        idx = list(itertools.combinations(range(N), 3))
        idx_arr = np.array(idx)
        worst_pts = set()
        for ti in worst_tris:
            for p in idx_arr[ti]:
                worst_pts.add(p)
        
        if not worst_pts:
            break
        
        improved = False
        step = base_step * (0.7 ** (it // 30))
        for pi in sorted(worst_pts):
            for di in range(2):  # angle or radial
                for delta in [-step, step]:
                    orig = params[pi, di]
                    params[pi, di] = orig + delta
                    pts = _params_to_points(params)
                    pts = _normalize(pts)
                    s = _score(pts)
                    if s > current + 1e-12:
                        current = s
                        improved = True
                        break
                    else:
                        params[pi, di] = orig
                if improved:
                    break
            if improved:
                break
        if not improved:
            if step < 1e-6:
                break
            base_step *= 0.5
    return params, current

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial configurations in parameter space
    param_configs = []
    
    # 1. Regular 14-gon (zero perturbation)
    param_configs.append(np.zeros((N, 2)))
    
    # 2. Small random angular/radial perturbations
    for scale in [0.02, 0.05, 0.1, 0.15]:
        param_configs.append(scale * np.random.randn(N, 2))
    
    # 3. Alternating radial pattern
    alt_radial = np.array([0.05 * ((-1)**i) for i in range(N)])
    param_configs.append(np.column_stack([np.zeros(N), alt_radial]))
    
    # 4. Quadratic radial modulation
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    quad_radial = 0.1 * np.cos(2 * angles)
    param_configs.append(np.column_stack([np.zeros(N), quad_radial]))
    
    # 5. Cubic angular modulation
    ang_mod = 0.08 * np.sin(3 * angles)
    param_configs.append(np.column_stack([ang_mod, np.zeros(N)]))
    
    # 6. Mixed perturbations
    for _ in range(5):
        ang_p = 0.1 * np.random.randn(N)
        rad_p = 0.1 * np.random.randn(N)
        param_configs.append(np.column_stack([ang_p, rad_p]))
    
    # Also generate some raw Cartesian configs for diversity
    cartesian_configs = []
    # Regular 14-gon
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    cartesian_configs.append(x0.copy())
    cartesian_configs.append(x0 + 0.03 * np.random.randn(N, 2))
    # Random in disk
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        th = np.random.rand(N) * 2 * np.pi
        cartesian_configs.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    
    # Optimize in parameter space (primary strategy)
    for p_init in param_configs:
        if time.time() > deadline - 25:
            break
        
        try:
            p_flat = p_init.flatten()
            
            # Phase 1: Smooth optimization in param space
            res1 = minimize(_smooth_objective_param, p_flat, args=(80.0,),
                          method='Powell',
                          options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline - 20:
                break
            
            p1 = res1.x.reshape(-1, 2)
            
            # Phase 2: Exact optimization in param space
            res2 = minimize(_neg_objective_param, p1.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline - 15:
                break
            
            p2 = res2.x.reshape(-1, 2)
            
            # Phase 3: Local refinement in param space
            if time.time() < deadline - 10:
                p3, s3 = _local_refine_param(p2, max_iters=200, base_step=0.005, deadline=deadline - 5)
            else:
                p3 = p2
                pts3 = _params_to_points(p3)
                pts3 = _normalize(pts3)
                s3 = _score(pts3)
            
            pts = _params_to_points(p3)
            pts = _normalize(pts)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # Also try raw Cartesian optimization for additional diversity
    for x_init in cartesian_configs:
        if time.time() > deadline - 15:
            break
        
        try:
            x_flat = x_init.flatten()
            
            res1 = minimize(_smooth_objective, x_flat, args=(80.0,),
                          method='Powell',
                          options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline - 10:
                break
            
            p1 = res1.x.reshape(-1, 2)
            p1 = _normalize(p1)
            
            res2 = minimize(_neg_objective, p1.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline - 8:
                break
            
            p2 = res2.x.reshape(-1, 2)
            p2 = _normalize(p2)
            
            if time.time() < deadline - 5:
                p3, s3 = _local_refine(p2, max_iters=150, base_step=0.005, deadline=deadline - 3)
            else:
                p3, s3 = p2, _score(p2)
            
            s = _score(p3)
            if s > best_score:
                best_score = s
                best_points = p3.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        best_points = _normalize(x0.copy())
    
    # Final global polish in param space
    if best_points is not None and time.time() < deadline - 5:
        try:
            params_final = _points_to_params(best_points)
            res_final = minimize(_neg_objective_param, params_final.flatten(),
                              method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            pf = _params_to_points(res_final.x.reshape(-1, 2))
            pf = _normalize(pf)
            sf = _score(pf)
            if sf > best_score:
                best_points = pf
                best_score = sf
        except Exception:
            pass
    
    return best_points
