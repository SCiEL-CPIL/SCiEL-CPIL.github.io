import numpy as np
import time
import scipy.spatial as spat
import itertools
from scipy.optimize import minimize

N = 13

def _get_triangle_areas(points):
    """
    Calculate areas of all triangles formed by the points.
    Returns an array of shape (num_triangles,).
    """
    n = len(points)
    # Pre-computed indices for C(13,3) = 286 triangles
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = points[j_flat] - points[i_flat] 
    v2 = points[k_flat] - points[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    return areas

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = spat.ConvexHull(points)
        return hull.volume # In 2D, volume is area
    except Exception:
        return 1e-9

def _score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _get_triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _softmin_score(points, beta):
    """
    Calculate a smooth approximation of the minimum triangle area divided by hull area.
    Returns the negative score to be used for minimization.
    """
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 1e9

    areas = _get_triangle_areas(points)
    
    # Softmin of areas
    # To avoid overflow in exp, shift by max
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        softmin_val = max_area
    else:
        softmin_val = max_area - (1.0 / beta) * np.log(sum_exp)
        
    # We want to maximize softmin_area / hull_area
    # So we minimize - (softmin_area / hull_area)
    return -(softmin_val / hull_area)

def _make_ring_config(radii, angles):
    """
    Construct points from ring radii and angles.
    radii: array of shape (N,)
    angles: array of shape (N,)
    """
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def _optimize_from_start(start_points, beta_schedule, max_iter=200, deadline=None):
    """
    Optimize the softmin score using L-BFGS-B.
    """
    current_points = start_points.copy()
    best_score = _score(current_points)
    
    for beta in beta_schedule:
        if deadline and time.time() > deadline:
            break
            
        # Define objective function for scipy
        def objective(x_flat):
            pts = x_flat.reshape(-1, 2)
            # Keep points centered to reduce numerical drift? 
            # Actually, the score is translation invariant, so we can fix the center of mass to 0 to reduce degrees of freedom?
            # No, let's just let it move, but we must ensure it doesn't collapse.
            # Adding a small penalty for very small hull area is handled in _softmin_score.
            return _softmin_score(pts, beta)
        
        # Initial guess
        x0 = current_points.flatten()
        
        # Run minimization
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < objective(x0):
                new_points = res.x.reshape(-1, 2)
                new_score = _score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    current_points = new_points
        except Exception:
            pass
            
    return current_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Define a schedule of beta values. Start low (smooth) to high (sharp).
    # The transition from smooth to sharp helps escape local minima of the smooth approximation.
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    # Generate diverse initial configurations
    initializations = []
    
    # 1. Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.ones(N)
    initializations.append(_make_ring_config(radii, angles))
    
    # 2. Fibonacci lattice (Sunflower pattern)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_radii = np.sqrt(np.arange(N) + 0.5) / np.sqrt(N)
    fib_angles = golden_angle * np.arange(N)
    initializations.append(_make_ring_config(fib_radii, fib_angles))
    
    # 3. Two rings: 1 center, 6 inner, 6 outer? Or 1-5-7?
    # Let's try 1 center, 5 inner, 7 outer
    radii_3 = np.array([0.0] + [0.5]*5 + [1.0]*7)
    angles_3 = np.concatenate([
        [0.0],
        np.linspace(0, 2*np.pi, 5, endpoint=False),
        np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7 # offset
    ])
    initializations.append(_make_ring_config(radii_3, angles_3))
    
    # 4. Random perturbation of regular polygon
    for i in range(5):
        noise_r = np.random.normal(0, 0.2, N)
        noise_a = np.random.normal(0, 0.2, N)
        r = np.ones(N) + noise_r
        a = angles + noise_a
        initializations.append(_make_ring_config(r, a))

    # Optimize each initialization
    for init_pts in initializations:
        if time.time() > deadline - 20.0:
            break
            
        # Center the points
        init_pts -= np.mean(init_pts, axis=0)
        
        # Run optimization
        polished_pts, score = _optimize_from_start(init_pts, beta_schedule, max_iter=100, deadline=deadline - 10.0)
        
        if score > best_score:
            best_score = score
            best_points = polished_pts
            
    # Final local polish on the best found
    if best_points is not None and time.time() < deadline - 5.0:
        # Use a very high beta for final sharpening
        final_pts, final_score = _optimize_from_start(best_points, [200.0, 500.0], max_iter=50, deadline=deadline - 2.0)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts

    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
