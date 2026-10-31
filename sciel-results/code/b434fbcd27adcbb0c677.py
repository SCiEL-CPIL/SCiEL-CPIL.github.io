import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import minimize

N = 13

# Pre-compute triangle indices
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0]):
            return 0.0
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _points_from_params(params):
    """
    params: (25,) array.
    First 12: alpha_1, ..., alpha_12 (angle differences)
    Next 13: d_1, ..., d_13 (distances)
    """
    alphas = params[:12]
    ds = params[12:]
    
    # Check validity
    if np.any(alphas <= 1e-6) or np.any(ds <= 1e-6):
        return None
        
    alpha_13 = 2 * np.pi - np.sum(alphas)
    if alpha_13 <= 1e-6:
        return None
        
    all_alphas = np.concatenate([alphas, [alpha_13]])
    
    # Compute theta
    thetas = np.cumsum(np.concatenate([[0], all_alphas]))[:-1]
    
    # Compute vertices
    points = np.zeros((N, 2))
    for i in range(N):
        theta_i = thetas[i]
        theta_next = thetas[(i + 1) % N]
        d_i = ds[i]
        d_next = ds[(i + 1) % N]
        
        # Solve for intersection
        # cos(theta_i) * x + sin(theta_i) * y = d_i
        # cos(theta_next) * x + sin(theta_next) * y = d_next
        
        a11 = np.cos(theta_i)
        a12 = np.sin(theta_i)
        a21 = np.cos(theta_next)
        a22 = np.sin(theta_next)
        
        det = a11 * a22 - a12 * a21
        if np.abs(det) < 1e-10:
            return None
            
        x = (d_i * a22 - a12 * d_next) / det
        y = (a11 * d_next - d_i * a21) / det
        
        points[i] = [x, y]
        
    return points

def _softmin_score_from_params(params, beta=50):
    points = _points_from_params(params)
    if points is None:
        return 1e10
        
    # Center the points
    points = points - np.mean(points, axis=0)
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
        
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Log-sum-exp based soft-min
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_areas - max_neg)))
    soft_min = -log_sum / beta
    
    score = soft_min / hull_area
    if score <= 0:
        return 1e10
    return -score

def _coordinate_descent_polish(points, deadline, max_iter=5000):
    """Coordinate descent on the hard objective: move each point's x and y independently."""
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations in this coordinate
                for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001]:
                    trial = best_pts.copy()
                    trial[i, coord] += delta
                    trial_s = _score(trial)
                    if trial_s > best_s + 1e-14:
                        best_pts = trial
                        best_s = trial_s
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_pts, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -1.0
    best_points = None
    
    # Initial parameters
    # Regular polygon: equal alphas, equal ds
    base_alpha = 2 * np.pi / N
    alphas = np.full(12, base_alpha)
    ds = np.full(N, 1.0)
    init_params = np.concatenate([alphas, ds])
    
    # Optimize
    try:
        res = minimize(
            _softmin_score_from_params,
            init_params,
            method='Powell',
            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        best_params = res.x
    except Exception:
        best_params = init_params
        
    points = _points_from_params(best_params)
    if points is not None:
        points = points - np.mean(points, axis=0)
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
            
    # Polish
    if best_points is not None and time.time() < deadline - 5.0:
        polished_pts, polished_score = _coordinate_descent_polish(best_points, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
            
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
