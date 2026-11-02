import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

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

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    return -_score(pts)

def _soft_min_objective(flat_points, beta=50.0):
    """
    Smooth surrogate for the min triangle area objective.
    """
    pts = flat_points.reshape(N, 2)
    pi = pts[IDX_I]
    pj = pts[IDX_J]
    pk = pts[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1.0
        
    m = np.min(areas)
    shifted = areas - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = m - (1.0 / beta) * log_sum
    
    score = soft_min_val / hull_area
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8, use_smooth=False):
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        obj_func = _soft_min_objective if use_smooth else _objective
        
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(obj_func, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(obj_func, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # 1. Initializations
    initializations = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(x_reg)
    
    # Grid 4x4 (remove 2)
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    x_grid = grid_pts[:14]
    initializations.append(x_grid)
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Random points
    rng = np.random.default_rng(123)
    for _ in range(2):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)

    # 2. Refine initializations with SMOOTH objective
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1500, use_smooth=True)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        if time.time() < deadline - 20.0:
            cand_pts2, cand_score2 = refine(cand_pts, 'Nelder-Mead', maxiter=500, use_smooth=False)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 3. Differential Evolution with SMOOTH objective
    if time.time() < deadline - 30.0:
        def de_objective(flat_points):
            return _soft_min_objective(flat_points)
            
        try:
            bounds = [(-1, 1)] * (N * 2)
            res_de = differential_evolution(
                de_objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10, 
                tol=1e-4, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                cand_pts = res_de.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
        except Exception:
            pass

    # 4. Final polish
    if best_points is not None and time.time() < deadline - 10.0:
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > deadline - 10.0:
                break
            cand_pts, cand_score = refine(best_points, method, maxiter=1000, use_smooth=False)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
