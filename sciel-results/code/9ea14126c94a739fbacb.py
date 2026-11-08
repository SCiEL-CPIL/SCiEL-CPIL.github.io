"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _triangle_areas(points):
    """Compute all triangle areas from a set of points."""
    n = len(points)
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas[i, j, k] = area
                areas[j, i, k] = area
                areas[k, j, i] = area
    return areas

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shoelace formula on sorted points
        # This is a rough fallback
        return 1.0

def _min_triangle_area(points):
    """Return the minimum triangle area among all triples."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_area = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _soft_min_triangle_area(points, beta):
    """Soft minimum of triangle areas using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # For numerical stability, shift by min
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted))) + beta * m
    soft_min = -log_sum_exp / beta
    return soft_min

def _soft_score(points, beta):
    """Soft score = soft_min_triangle_area / convex_hull_area."""
    min_area = _soft_min_triangle_area(points, beta)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_area / hull_area

def _neg_soft_score(points, beta):
    return -_soft_score(points, beta)

def _gradient_neg_soft_score(points, beta):
    """Numerical gradient of the negative soft score."""
    n = len(points)
    grad = np.zeros_like(points)
    eps = 1e-7
    base = _neg_soft_score(points, beta)
    for i in range(n):
        for d in range(2):
            orig = points[i, d]
            points[i, d] = orig + eps
            f_plus = _neg_soft_score(points, beta)
            points[i, d] = orig - eps
            f_minus = _neg_soft_score(points, beta)
            points[i, d] = orig
            grad[i, d] = (f_plus - f_minus) / (2 * eps)
    return grad

def _random_initial(seed=42):
    rng = np.random.RandomState(seed)
    points = rng.rand(N, 2)
    return points

def _regular_polygon(n, seed=0):
    """Create a regular n-gon centered at origin with unit area."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # For a regular n-gon with unit area, we need to scale appropriately
    # Area of regular n-gon with circumradius R: (n/2) * R^2 * sin(2*pi/n)
    # Set area = 1: R = sqrt(2 / (n * sin(2*pi/n)))
    R = math.sqrt(2.0 / (n * math.sin(2.0 * math.pi / n)))
    points = np.column_stack([R * np.cos(angles), R * np.sin(angles)])
    return points

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Try several initial configurations
    initial_configs = []
    
    # Regular polygon
    initial_configs.append(_regular_polygon(N))
    
    # Random configurations with different seeds
    for seed in [42, 123, 7, 999, 2024]:
        initial_configs.append(_random_initial(seed))
    
    # Add some structured configurations
    # Two concentric regular polygons (7+7)
    inner_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    R_inner = 0.5
    R_outer = 1.0
    inner_pts = np.column_stack([R_inner * np.cos(inner_angles), R_inner * np.sin(inner_angles)])
    outer_pts = np.column_stack([R_outer * np.cos(outer_angles), R_outer * np.sin(outer_angles)])
    initial_configs.append(np.vstack([inner_pts, outer_pts]))
    
    # Grid-like configuration
    grid_x = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.7, 0.5, 0.1, 0.9, 0.45, 0.55, 0.35, 0.65, 0.5])
    grid_y = np.array([0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.35, 0.65, 0.6, 0.4, 0.5])
    initial_configs.append(np.column_stack([grid_x, grid_y]))
    
    # Optimize each initial configuration
    for init_idx, points in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Multi-beta annealing: start with low beta (smooth), increase to high beta (sharp)
        betas = [1.0, 5.0, 20.0, 50.0, 100.0]
        
        current_points = points.copy()
        
        for beta in betas:
            if time.time() > deadline:
                break
            
            # Use L-BFGS-B with numerical gradient
            result = minimize(
                _neg_soft_score,
                current_points,
                args=(beta,),
                jac=_gradient_neg_soft_score,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < _neg_soft_score(current_points, beta):
                current_points = result.x
            
            # Evaluate actual score
            score = _score(current_points)
            if score > best_score:
                best_score = score
                best_points = current_points.copy()
        
        # Final polish with high beta and tighter tolerance
        if time.time() < deadline - 5.0:
            for beta in [100.0, 200.0, 500.0]:
                if time.time() > deadline - 2.0:
                    break
                
                result = minimize(
                    _neg_soft_score,
                    current_points,
                    args=(beta,),
                    jac=_gradient_neg_soft_score,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < _neg_soft_score(current_points, beta):
                    current_points = result.x
                
                score = _score(current_points)
                if score > best_score:
                    best_score = score
                    best_points = current_points.copy()
    
    # If we haven't found a good solution, try differential evolution
    if best_score < 0.01 and time.time() < deadline - 30.0:
        def _neg_score_de(x):
            points = x.reshape(N, 2)
            return -_score(points)
        
        bounds = [(0, 1)] * (2 * N)
        result_de = differential_evolution(
            _neg_score_de,
            bounds=bounds,
            seed=42,
            maxiter=100,
            tol=1e-6,
            polish=True,
            init='sobol',
            workers=1
        )
        
        if result_de.success:
            points_de = result_de.x.reshape(N, 2)
            score_de = _score(points_de)
            if score_de > best_score:
                best_score = score_de
                best_points = points_de.copy()
    
    # Final check: if best_points is None, return a regular polygon
    if best_points is None:
        best_points = _regular_polygon(N)
    
    # Ensure points are within [0,1] x [0,1]
    best_points = np.clip(best_points, 0.0, 1.0)
    
    return best_points
# EVOLVE-BLOCK-END
