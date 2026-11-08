import numpy as np
import time
import itertools
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 145.0  # Safety margin before 150s kill

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
    shifted = np.clip(shifted, 0, 50.0 / tau)  # numerical stability
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

def _params_to_points(params):
    """Convert angular/radial perturbation params to points on a convex curve.
    params: (N, 2) array where [:, 0] is angular offset, [:, 1] is radial scale.
    Base: regular 14-gon. Angles are base_angle + param[:,0], radii are 1.0 * param[:,1].
    """
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + params[:, 0]
    radii = np.maximum(params[:, 1], 0.1)  # ensure positive radius
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _neg_objective(flat):
    """Negative min triangle area after normalization (for Cartesian points)."""
    pts = flat.reshape(-1, 2)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _neg_objective_param(flat):
    """Negative score for parameterized points (angular/radial perturbations)."""
    params = flat.reshape(N, 2)
    pts = _params_to_points(params)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    return -np.min(areas)

def _smooth_objective_param(flat, tau=80.0):
    """Smooth surrogate for parameterized points."""
    params = flat.reshape(N, 2)
    pts = _params_to_points(params)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    sm = _soft_min(areas, tau)
    return -sm / ha

def _local_refine(pts, max_iters=200, base_step=0.005, deadline=None):
    """Fast local refinement on Cartesian coordinates."""
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

def _local_refine_param(params, max_iters=200, base_step=0.01, deadline=None):
    """Local refinement on parameter space (angular/radial)."""
    params = params.copy()
    pts = _params_to_points(params)
    current = _score(_normalize(pts))
    n = N
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        pts = _params_to_points(params)
        pts_norm = _normalize(pts)
        areas = _triangle_areas(pts_norm)
        min_a = np.min(areas)
        tol = 1e-6
        worst_mask = np.abs(areas - min_a) < tol
        worst_tris = np.where(worst_mask)[0][:30]
        worst_pts = set()
        for ti in worst_tris:
            for p in np.array(list(itertools.combinations(range(n), 3)))[ti]:
                worst_pts.add(p)
        
        if not worst_pts:
            break
        
        improved = False
        step = base_step * (0.7 ** (it // 20))
        for pi in sorted(worst_pts):
            # Perturb angular and radial params
            for d_ang, d_rad in [(-step, 0), (step, 0), (0, -step), (0, step),
                                 (-step, -step), (step, step), (-step, step), (step, -step)]:
                orig_a, orig_r = params[pi, 0], params[pi, 1]
                params[pi, 0] += d_ang
                params[pi, 1] = max(params[pi, 1] + d_rad, 0.1)
                pts = _params_to_points(params)
                s = _score(_normalize(pts))
                if s > current + 1e-12:
                    current = s
                    improved = True
                    break
                else:
                    params[pi, 0] = orig_a
                    params[pi, 1] = orig_r
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
    
    # Generate diverse initial parameter configurations
    param_configs = []
    
    # 1. Regular 14-gon (zero perturbations)
    param_configs.append(np.zeros((N, 2)))
    
    # 2. Small random angular perturbations
    for scale in [0.02, 0.05, 0.1]:
        param_configs.append(np.column_stack([
            scale * np.random.randn(N),
            np.ones(N) + 0.02 * np.random.randn(N)
        ]))
    
    # 3. Small random radial perturbations
    for scale in [0.05, 0.1, 0.15]:
        param_configs.append(np.column_stack([
            0.02 * np.random.randn(N),
            np.ones(N) + scale * np.random.randn(N)
        ]))
    
    # 4. Mixed perturbations
    for _ in range(3):
        param_configs.append(np.column_stack([
            0.05 * np.random.randn(N),
            np.ones(N) + 0.08 * np.random.randn(N)
        ]))
    
    # 5. Systematic angular shifts (rotational symmetry breaking)
    for shift in [0.1, 0.2, 0.3]:
        params = np.zeros((N, 2))
        params[:, 0] = shift * np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
        param_configs.append(params)
    
    # 6. Systematic radial variations
    for _ in range(3):
        params = np.zeros((N, 2))
        params[:, 1] = 1.0 + 0.1 * np.cos(np.linspace(0, 4*np.pi, N, endpoint=False))
        param_configs.append(params)
    
    for p_init in param_configs:
        if time.time() > deadline - 25:
            break
        
        try:
            p_flat = p_init.flatten()
            
            # Phase 1: Smooth optimization in parameter space with Powell
            res1 = minimize(_smooth_objective_param, p_flat, args=(80.0,),
                          method='Powell',
                          options={'maxiter': 300, 'xtol': 1e-8, 'ftol': 1e-10, 'adaptive': True})
            
            if time.time() > deadline - 15:
                break
            
            p1 = res1.x.reshape(N, 2)
            
            # Phase 2: Exact optimization in parameter space with Nelder-Mead
            res2 = minimize(_neg_objective_param, p1.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            if time.time() > deadline - 10:
                break
            
            p2 = res2.x.reshape(N, 2)
            pts2 = _params_to_points(p2)
            pts2 = _normalize(pts2)
            
            # Phase 3: Local refinement in parameter space
            if time.time() < deadline - 8:
                p3, s3 = _local_refine_param(p2, max_iters=150, base_step=0.005, deadline=deadline - 5)
                pts3 = _params_to_points(p3)
                pts3 = _normalize(pts3)
            else:
                p3, s3 = p2, _score(pts2)
                pts3 = pts2
            
            s = _score(pts3)
            if s > best_score:
                best_score = s
                best_points = pts3.copy()
                
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_points = _normalize(best_points)
    
    # Final polish in Cartesian space for maximum precision
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_final = minimize(_neg_objective, best_points.flatten(),
                              method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            pf = res_final.x.reshape(-1, 2)
            pf = _normalize(pf)
            sf = _score(pf)
            if sf > best_score:
                best_points = pf
                best_score = sf
        except Exception:
            pass
    
    return best_points
