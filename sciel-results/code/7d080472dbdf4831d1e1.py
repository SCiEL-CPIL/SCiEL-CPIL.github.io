import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

# Precompute indices for all triples
_triples = np.array(list(itertools.combinations(range(N), 3)))
_idx_i = _triples[:, 0]
_idx_j = _triples[:, 1]
_idx_k = _triples[:, 2]

def _compute_areas(points):
    """
    Compute areas of all triangles formed by triples of points.
    points: (N, 2) array
    Returns: array of length C(N,3)
    """
    p_i = points[_idx_i]
    p_j = points[_idx_j]
    p_k = points[_idx_k]
    
    # Vectorized cross product
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
        areas = _compute_areas(points)
        min_area = np.min(areas)
        return min_area / hull_area
    except Exception:
        return 0.0

def _objective(x, beta):
    """
    Negative soft-min of the triangle area ratios.
    We want to maximize the min ratio, so we minimize the negative soft-min.
    """
    points = x.reshape(-1, 2)
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 1e10
        areas = _compute_areas(points)
        ratios = areas / hull_area
        
        # Soft min: -1/beta * log(sum(exp(-beta * ratios)))
        # Use log-sum-exp for numerical stability
        max_r = np.max(ratios)
        # exp(-beta * (ratios - max_r))
        exp_terms = np.exp(-beta * (ratios - max_r))
        log_sum = np.log(np.sum(exp_terms))
        log_sum_exp = log_sum - beta * max_r
        soft_min = -1.0 / beta * log_sum_exp
        
        return -soft_min
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin

    # Generate diverse initial candidates
    candidates = []
    rng = np.random.default_rng(42)

    # 1. Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Sunflower pattern (golden angle)
    golden_angle = np.pi * (3. - np.sqrt(5.))
    angles_sf = np.arange(N) * golden_angle
    radii_sf = np.sqrt(np.arange(1, N + 1)) / np.sqrt(N)
    candidates.append(np.column_stack([radii_sf * np.cos(angles_sf), radii_sf * np.sin(angles_sf)]))

    # 3. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        candidates.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    # 4. Grid-like (4x4 minus 3)
    # Create a 4x4 grid, remove 3 points to get 13
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_points = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    # Remove 3 points (e.g., corners or random)
    idx_to_remove = rng.choice(16, 3, replace=False)
    mask = np.ones(16, dtype=bool)
    mask[idx_to_remove] = False
    candidates.append(grid_points[mask])

    # 5. Circle with noise
    for _ in range(3):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        noise = rng.uniform(-0.2, 0.2, N)
        radii = 1.0 + noise
        candidates.append(np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]))

    best_score = 0.0
    best_points = candidates[0]

    # Optimization settings
    beta = 30.0  # Softness parameter

    for i, x0 in enumerate(candidates):
        if time.time() > deadline:
            break

        def obj(x):
            return _objective(x, beta)

        try:
            # L-BFGS-B is robust and fast
            res = minimize(obj, x0.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-6})
            
            if res.fun < 1e9:  # Check if it didn't fail badly
                score = _score(res.x.reshape(-1, 2))
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(-1, 2)
        except Exception:
            pass

    # Final polish with higher beta for better precision
    if time.time() < deadline - 5.0:
        beta_high = 100.0

        def obj_high(x):
            return _objective(x, beta_high)

        try:
            res = minimize(obj_high, best_points.flatten(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7})
            if res.fun < 1e9:
                score = _score(res.x.reshape(-1, 2))
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(-1, 2)
        except Exception:
            pass

    return best_points
