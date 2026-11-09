import time
import numpy as np
from scipy.spatial import ConvexHull
import itertools
from scipy.optimize import minimize

_DEADLINE = time.time() + 585.0  # Safety margin

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _ratio(points):
    """Compute min_triangle_area / hull_area. Returns 0 if invalid."""
    if not np.all(np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma < 1e-12:
        return 0.0
    return ma / ha

def _neg_ratio(params):
    """Objective: minimize -ratio. Scale-invariant, so no normalization needed."""
    if time.time() > _DEADLINE:
        return 1e10
    points = params.reshape((13, 2))
    r = _ratio(points)
    if r < 1e-12:
        return 1e10
    return -r

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_circles(n_inner, n_outer, r_inner=0.3, r_outer=1.0):
    """Points on two concentric circles."""
    inner = _regular_polygon(n_inner, radius=r_inner)
    outer = _regular_polygon(n_outer, radius=r_outer)
    return np.vstack([inner, outer])

def _grid_like(n):
    """Grid-like arrangement for 13 points."""
    # 4x4 grid minus one corner = 15, use 13
    xs = np.array([0, 1, 2, 3])
    ys = np.array([0, 1, 2, 3])
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    # Remove 2 points to get 13
    pts = pts[:13]
    # Center and scale
    pts = pts - pts.mean(axis=0)
    return pts

def _random_points(n, rng, scale=1.0):
    return rng.uniform(-scale, scale, size=(n, 2))

def heilbronn_convex13() -> np.ndarray:
    n = 13
    best_points = None
    best_score = -np.inf
    
    rng = np.random.default_rng(seed=42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular 13-gon
    initial_configs.append(_regular_polygon(n, radius=1.0))
    
    # Concentric circles: 7 inner + 6 outer
    initial_configs.append(_concentric_circles(7, 6, r_inner=0.4, r_outer=1.0))
    
    # Concentric circles: 5 inner + 8 outer
    initial_configs.append(_concentric_circles(5, 8, r_inner=0.35, r_outer=1.0))
    
    # Concentric circles: 3 inner + 10 outer
    initial_configs.append(_concentric_circles(3, 10, r_inner=0.3, r_outer=1.0))
    
    # Grid-like
    initial_configs.append(_grid_like(n))
    
    # Random configurations
    for _ in range(5):
        initial_configs.append(_random_points(n, rng, scale=1.0))
    
    # Perturbed regular polygon
    for _ in range(3):
        perturbed = _regular_polygon(n, radius=1.0) + rng.normal(0, 0.1, (n, 2))
        initial_configs.append(perturbed)
    
    # Evaluate each initial config and run local optimization
    for config in initial_configs:
        if time.time() > _DEADLINE - 60:
            break
        
        # Evaluate initial score
        init_score = _ratio(config)
        if init_score > best_score:
            best_score = init_score
            best_points = config.copy()
        
        # Run Nelder-Mead optimization
        try:
            params = config.flatten()
            res = minimize(_neg_ratio, params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10,
                                   'adaptive': True})
            
            opt_points = res.x.reshape((n, 2))
            opt_score = _ratio(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
                
        except Exception:
            continue
    
    # If still no good result, try more random restarts with Nelder-Mead
    if best_score < 0.01:
        for trial in range(20):
            if time.time() > _DEADLINE - 30:
                break
            
            # Random start
            start = _random_points(n, rng, scale=1.0)
            
            try:
                params = start.flatten()
                res = minimize(_neg_ratio, params, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10,
                                       'adaptive': True})
                
                opt_points = res.x.reshape((n, 2))
                opt_score = _ratio(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
            except Exception:
                continue
    
    # Fallback to regular polygon
    if best_points is None or best_score < 0.001:
        best_points = _regular_polygon(n, radius=1.0)
        best_score = _ratio(best_points)
    
    # Final check: Ensure points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(n, radius=1.0)
        
    return best_points
