import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar

N = 14

# Pre-compute indices for all triangles i < j < k
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _two_heptagons(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    with inner ring rotated by theta relative to outer ring.
    """
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def _score_two_heptagons(r, theta):
    """Score for two heptagons configuration."""
    pts = _two_heptagons(r, theta)
    return _score(pts)

def _neg_score_two_heptagons(params):
    """Negative score for optimization."""
    r, theta = params
    if r <= 0 or r > 1:
        return 1.0
    return -_score_two_heptagons(r, theta)

def _local_optimize_free(points, time_limit=10.0):
    """Local optimization of all points freely using Powell's method."""
    start_time = time.time()
    
    def obj(x):
        if time.time() > start_time + time_limit:
            return 1e6  # Penalize timeout
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = points.flatten()
    
    try:
        res = minimize(
            obj, x0, method='Powell',
            options={
                'maxiter': 80, 
                'xtol': 1e-9, 
                'ftol': 1e-11
            }
        )
        
        if time.time() < start_time + time_limit:
            final_score = -res.fun
            if final_score > 0:
                return final_score, res.x.reshape(-1, 2).copy()
                
    except Exception:
        pass
    
    return None

def _refine_with_sa(points, time_limit=15.0, n_restarts=3):
    """
    Refine points using simulated annealing with multiple restarts.
    Returns the best points found.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(best_points)
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > start_time + time_limit - 2:
            break
        
        current_points = best_points.copy()
        current_score = best_score
        
        T_start = 0.005
        T_end = 1e-7
        n_steps = 800
        step_scale = 0.015
        
        for i in range(n_steps):
            if time.time() > start_time + time_limit - 2:
                break
            
            noise = rng.normal(0, step_scale, size=(N, 2))
            new_points = current_points + noise
            
            new_score = _score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                delta = current_score - new_score
                if delta > 0:
                    T = T_start * (T_end / T_start) ** (i / n_steps)
                    if T > 1e-10:
                        prob = np.exp(-delta / T)
                        if rng.random() < prob:
                            current_points = new_points
                            current_score = new_score
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: Two concentric heptagons
    # Based on previous best practices, r ~ 0.5-0.7, theta ~ pi/7
    best_points = _two_heptagons(0.6, np.pi / 7)
    best_score = _score(best_points)
    
    # Strategy 1: 2D grid search over (r, theta)
    # We'll do a coarse grid search first, then refine.
    
    r_values = np.linspace(0.3, 0.9, 13)
    theta_values = np.linspace(0, 2 * np.pi / 7, 14, endpoint=False)
    
    for r in r_values:
        if time.time() > deadline - 30:
            break
        for theta in theta_values:
            if time.time() > deadline - 30:
                break
            score = _score_two_heptagons(r, theta)
            if score > best_score:
                best_score = score
                best_points = _two_heptagons(r, theta)
    
    # Strategy 2: Local optimization of (r, theta) using Powell
    # We'll run this from multiple starting points
    
    start_points = [
        (0.5, np.pi / 7),
        (0.6, np.pi / 7),
        (0.7, np.pi / 7),
        (0.5, 0),
        (0.6, 0),
        (0.7, 0),
        (0.55, np.pi / 14),
        (0.65, 3 * np.pi / 14),
    ]
    
    for r0, theta0 in start_points:
        if time.time() > deadline - 20:
            break
        
        x0 = np.array([r0, theta0])
        
        try:
            res = minimize(
                _neg_score_two_heptagons, x0, method='Powell',
                bounds=[(0.01, 0.99), (0, 2 * np.pi / 7)],
                options={
                    'maxiter': 60, 
                    'xtol': 1e-9, 
                    'ftol': 1e-11
                }
            )
            
            if time.time() < deadline - 10:
                r_opt, theta_opt = res.x
                if 0.01 < r_opt < 0.99:
                    score = _score_two_heptagons(r_opt, theta_opt)
                    if score > best_score:
                        best_score = score
                        best_points = _two_heptagons(r_opt, theta_opt)
                        
        except Exception:
            pass
    
    # Strategy 3: Free local optimization of all points
    # This allows the points to move away from the strict two-heptagon structure
    
    if time.time() < deadline - 15:
        result = _local_optimize_free(best_points, time_limit=12.0)
        if result is not None:
            score, points = result
            if score > best_score:
                best_score = score
                best_points = points
    
    # Strategy 4: Simulated Annealing for final polish
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        best_points, best_score = _refine_with_sa(best_points, time_limit=remaining_time, n_restarts=2)
    
    # Fallback if something went wrong
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
