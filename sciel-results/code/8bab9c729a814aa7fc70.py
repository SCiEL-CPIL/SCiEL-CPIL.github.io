"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time


def _min_triangle_area_normalized(points):
    """
    Compute the minimum area of any triangle formed by 3 points,
    normalized by the convex hull area.
    
    points: (N, 2) array
    Returns: float, the normalized min triangle area (higher is better)
    """
    n = len(points)
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if hull fails, use a large penalty
        return 0.0
    
    if hull_area <= 1e-12:
        return 0.0
    
    # Find minimum triangle area
    # Vectorized approach for speed
    # Generate indices
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    
    # Mask for i < j < k
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Cross product in 2D: v1[0]*v2[1] - v1[1]*v2[0]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area


def _objective(params):
    """
    Objective function for optimization: minimize negative of min normalized triangle area.
    
    params: (N,) array of angles (radians) on a circle
    Returns: float, negative of the normalized min triangle area
    """
    # Convert angles to points on unit circle
    angles = params
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    score = _min_triangle_area_normalized(points)
    return -score


def construct():
    """
    Construct N points in a planar convex region to maximize the minimum
    triangle area normalized by convex hull area.
    
    Returns: (N, 2) array of points
    """
    best_score = -np.inf
    best_points = None
    
    # Generate multiple initial guesses
    initial_guesses = []
    
    # 1. Regular polygon
    initial_guesses.append(np.linspace(0, 2 * np.pi, N, endpoint=False))
    
    # 2. Random perturbations of regular polygon
    rng = np.random.default_rng(42)
    for _ in range(10):
        noise = rng.normal(0, 0.1, N)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + noise
        # Sort angles to maintain convex order? 
        # Nelder-Mead doesn't strictly require sorted angles if we just map to circle,
        # but the "convex position" implies we should consider the cyclic order.
        # However, the objective function is permutation invariant in terms of the set of points,
        # but the parameterization `cos, sin` maps angles to points.
        # If angles are not sorted, the points are just a set.
        # The convex hull calculation handles the set.
        # So we don't strictly need to sort, but it helps local search if we think of it as a polygon.
        initial_guesses.append(angles)
        
    # 3. Some random configurations
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        initial_guesses.append(angles)
        
    # Optimize each guess
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    for init_angles in initial_guesses:
        if time.time() - start_time > timeout:
            break
            
        try:
            result = minimize(
                _objective,
                init_angles,
                method='Nelder-Mead',
                options={
                    'maxiter': 500, # Reduced maxiter per run to allow more starts
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_points = np.column_stack([np.cos(result.x), np.sin(result.x)])
                
        except Exception:
            continue
            
    if best_points is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points
# EVOLVE-BLOCK-END
