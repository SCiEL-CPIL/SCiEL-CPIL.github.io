import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14 to speed up area calculation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _soft_min_score(points, beta=100.0):
    """
    Soft-min surrogate for the minimum triangle area.
    Uses log-sum-exp for numerical stability.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)

    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    return soft_min / hull_area

def _objective_flat(x, beta=100.0):
    """
    Objective function for flat array optimization.
    x: (2N,) array
    Returns negative soft-min score to be minimized.
    """
    points = x.reshape(N, 2)
    score = _soft_min_score(points, beta)
    return -score

def _construct_direct_optimization(deadline):
    """
    Structural change: Direct optimization of 2N coordinates.
    1. DE to find a good basin.
    2. L-BFGS-B with annealed beta to sharpen.
    """
    start_time = time.time()
    
    # Bounds for coordinates: [-1, 1] is sufficient since we normalize by hull area
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    # Initial guess: Random points in unit square
    x0 = np.random.uniform(-0.5, 0.5, 2 * N)
    
    # Phase 1: Differential Evolution
    # Use a relatively low beta for DE to provide smoother gradients
    beta_de = 20.0
    current_x = x0
    
    if time.time() < deadline - 30:
        try:
            res_de = differential_evolution(
                _objective_flat, 
                bounds, 
                args=(beta_de,),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-5,
                polish=False # We will do our own polish
            )
            if res_de.fun < -0.0: # Check if valid improvement
                current_x = res_de.x
        except Exception:
            pass

    # Phase 2: Gradient-based refinement with increasing beta
    # Start with moderate beta and increase to sharpen the minimum
    betas = [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        try:
            res_lbfgs = minimize(
                _objective_flat,
                current_x,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000}
            )
            if -res_lbfgs.fun > 0: # Valid improvement
                current_x = res_lbfgs.x
        except Exception:
            pass

    points = current_x.reshape(N, 2)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_score = 0.0
    best_points = None
    
    # Strategy 1: Direct Coordinate Optimization (Structural Change)
    if time.time() < deadline - 20:
        points_direct = _construct_direct_optimization(deadline)
        score_direct = _calculate_score(points_direct)
        if score_direct > best_score:
            best_score = score_direct
            best_points = points_direct

    # Strategy 2: Ring-based search as a fallback or complementary search
    # (Retained from original code but simplified to save time if direct works well)
    if time.time() < deadline - 5:
        # Quick check of a good ring configuration
        # 7-7 ring with optimized radius
        n_out, n_in = 7, 7
        # Use a known good radius from previous runs or quick grid
        for r in [0.5, 0.6, 0.7]:
            angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
            x_out = np.cos(angles_out)
            y_out = np.sin(angles_out)
            
            angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
            x_in = r * np.cos(angles_in)
            y_in = r * np.sin(angles_in)
            
            pts = np.vstack([np.column_stack((x_out, y_out)), np.column_stack((x_in, y_in))])
            s = _calculate_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts

    # Fallback if no valid points found
    if best_points is None:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
