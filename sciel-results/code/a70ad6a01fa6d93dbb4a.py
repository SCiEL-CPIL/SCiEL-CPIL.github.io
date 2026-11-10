import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

def _get_triangle_indices(n):
    """Precompute indices for all triangles."""
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                indices.append((i, j, k))
    return np.array(indices)

def _triangle_areas(points, indices):
    """
    Compute areas of all triangles defined by indices.
    points: (N, 2) array
    indices: (M, 3) array of (i, j, k)
    Returns: (M,) array of areas
    """
    p_i = points[indices[:, 0]]
    p_j = points[indices[:, 1]]
    p_k = points[indices[:, 2]]
    
    # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
    # Vectorized calculation
    x_i, y_i = p_i[:, 0], p_i[:, 1]
    x_j, y_j = p_j[:, 0], p_j[:, 1]
    x_k, y_k = p_k[:, 0], p_k[:, 1]
    
    areas = 0.5 * np.abs(x_i * (y_j - y_k) + x_j * (y_k - y_i) + x_k * (y_i - y_j))
    return areas

def _soft_min(areas, beta):
    """
    Smooth approximation of the minimum.
    As beta -> infinity, this approaches min(areas).
    For large beta, it is differentiable and sensitive to the smallest values.
    """
    # To avoid overflow in exp, subtract max
    max_area = np.max(areas)
    # Soft min is -log-sum-exp(-areas/beta) * beta
    # log-sum-exp(x) = max(x) + log(sum(exp(x - max(x))))
    # So soft_min = -beta * (max(-areas/beta) + log(sum(exp(-areas/beta - max(-areas/beta)))))
    # Let z = -areas/beta. max(z) = -min(areas)/beta.
    # This can be numerically unstable if areas are large or beta is small.
    # Simpler formulation:
    # soft_min = (1/beta) * log( sum( exp(-beta * areas) ) ) is for min? No.
    # min(x) = lim_{beta->inf} (1/beta) log( sum exp(-beta x) )
    # Let's use the stable log-sum-exp form.
    z = -beta * areas
    max_z = np.max(z)
    sum_exp = np.sum(np.exp(z - max_z))
    if sum_exp == 0:
        return 0.0
    return (max_z + np.log(sum_exp)) / beta

def _hull_area(points):
    """Calculate convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9 # Return small value to penalize invalid hulls

def _objective(params, n, indices, beta):
    """
    Objective function to minimize.
    We want to maximize A_min / A_hull.
    So we minimize - (A_min_soft / A_hull).
    """
    points = params.reshape(-1, 2)
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 1e10
        
    areas = _triangle_areas(points, indices)
    
    # If any area is 0, the ratio is 0. Soft min will be near 0.
    a_min_soft = _soft_min(areas, beta)
    
    a_hull = _hull_area(points)
    if a_hull < 1e-9:
        return 1e10
        
    ratio = a_min_soft / a_hull
    return -ratio

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize A_min / A_hull.
    Uses smooth soft-min surrogate and L-BFGS-B optimization on all 26 coordinates.
    """
    start_time = time.time()
    deadline = start_time + 550.0  # Safety margin for 600s limit
    
    n = 13
    indices = _get_triangle_indices(n)
    
    best_ratio = 0.0
    best_points = None
    
    # Initial guess: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    regular_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Evaluate initial guess
    areas = _triangle_areas(regular_points, indices)
    a_min = np.min(areas)
    a_hull = _hull_area(regular_points)
    best_ratio = a_min / a_hull
    best_points = regular_points.copy()
    
    # Optimization settings
    # Beta controls the "sharpness" of the soft-min. 
    # Larger beta -> closer to true min, but harder to optimize.
    # We will start with a moderate beta and potentially increase it.
    beta_start = 10.0
    
    # Multiple restarts with random perturbations
    rng = np.random.default_rng(42)
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Create initial guess: regular polygon + small random noise
        # Or random points in unit square? 
        # Regular polygon is a good baseline. Let's try random convex points.
        if i == 0:
            x0 = regular_points.flatten()
        else:
            # Random points in a square, then convex hull? 
            # Better: Random points on a circle with noise, or just random in square.
            # Random in square often leads to non-convex hulls or bad distributions.
            # Let's stick to perturbed regular polygon or random convex polygon.
            
            # Generate random convex polygon by random angles and radii
            angles_rand = np.sort(rng.random(n) * 2 * np.pi)
            # Ensure no two angles are too close
            radii_rand = 0.5 + 0.5 * rng.random(n)
            x_rand = radii_rand * np.cos(angles_rand)
            y_rand = radii_rand * np.sin(angles_rand)
            x0 = np.column_stack([x_rand, y_rand]).flatten()
            
            # Also try just regular polygon with different noise
            # x0 = regular_points.flatten() + rng.normal(0, 0.1, 2*n)
            
        # Refine beta: start low, increase later? 
        # For simplicity, use a fixed moderate beta for all restarts, 
        # but maybe increase beta in later iterations if time permits.
        beta = beta_start + i * 0.5
        
        try:
            result = minimize(
                _objective, 
                x0, 
                args=(n, indices, beta),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_ratio:
                points = result.x.reshape(-1, 2)
                # Evaluate true ratio
                areas = _triangle_areas(points, indices)
                a_min = np.min(areas)
                a_hull = _hull_area(points)
                
                if a_hull > 1e-9 and a_min > 0:
                    ratio = a_min / a_hull
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = points.copy()
                        
        except Exception:
            continue
            
    # Final local refinement using the best point found
    # Use a higher beta to sharpen the minimum
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        beta_final = 50.0
        try:
            result = minimize(
                _objective, 
                x0, 
                args=(n, indices, beta_final),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            points = result.x.reshape(-1, 2)
            areas = _triangle_areas(points, indices)
            a_min = np.min(areas)
            a_hull = _hull_area(points)
            if a_hull > 1e-9 and a_min > 0:
                ratio = a_min / a_hull
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()
        except Exception:
            pass

    return best_points
