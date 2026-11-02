import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for triangle areas
_i, _j, _k = np.indices((N, N, N))
_mask = (_i < _j) & (_j < _k)
_IDX_I = _i[_mask]
_IDX_J = _j[_mask]
_IDX_K = _k[_mask]

def _compute_min_triangle_area(points):
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    return np.min(areas)

def _compute_hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _objective(points_flat):
    points = points_flat.reshape(N, 2)
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    scale = np.sqrt(1.0 / hull_area)
    points = points * scale
    
    min_area = _compute_min_triangle_area(points)
    return -min_area

def _soft_min_objective(points_flat, beta):
    points = points_flat.reshape(N, 2)
    hull_area = _compute_hull_area(points)
    if hull_area <= 1e-10:
        return 1e10
    scale = np.sqrt(1.0 / hull_area)
    points = points * scale
    
    p_i = points[_IDX_I]
    p_j = points[_IDX_J]
    p_k = points[_IDX_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    soft_min = -max_area - (1.0/beta) * np.log(np.sum(exp_terms))
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    inits = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Jittered regular 14-gon
    for jitter_std in [0.05, 0.1, 0.2]:
        jitter = rng.normal(0, jitter_std, (N, 2))
        inits.append(np.column_stack([np.cos(angles) + jitter[:,0], np.sin(angles) + jitter[:,1]]))
        
    # 3. Random points in circle
    for _ in range(10):
        pts = rng.random((N, 2)) * 2 - 1
        dists = np.linalg.norm(pts, axis=1)
        max_dist = np.max(dists)
        if max_dist > 0:
            pts = pts / max_dist
        inits.append(pts)

    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        # Step 1: Coarse optimization with Powell
        try:
            res = minimize(_objective, x0, method='Powell', options={'maxiter': 500, 'maxfev': 5000})
            if time.time() > deadline:
                break
            x_current = res.x
        except Exception:
            x_current = x0

        # Step 2: Polish with Soft-Min L-BFGS-B
        # Anneal beta
        betas = [10, 50, 100, 500, 1000]
        for beta in betas:
            if time.time() > deadline:
                break
            try:
                res = minimize(lambda x: _soft_min_objective(x, beta), x_current, method='L-BFGS-B', options={'maxiter': 100, 'maxfev': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
                if time.time() > deadline:
                    break
                x_current = res.x
            except Exception:
                pass

        # Evaluate final score
        pts = x_current.reshape(N, 2)
        score = _score(pts)
        
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        best_points = np.array([[rng.random(), rng.random()] for _ in range(N)])
        
    return best_points
