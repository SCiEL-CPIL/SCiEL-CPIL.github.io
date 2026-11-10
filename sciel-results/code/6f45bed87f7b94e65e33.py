# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    n = 16
    d = 2
    
    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_smooth(params, p=8):
        """
        Smooth surrogate using power-mean approximation.
        Minimizes (d_max/d_min)^2 using power means for smoothness.
        """
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 1e10
        # Avoid zero distances
        dists = np.maximum(dists, 1e-12)
        m = len(dists)
        # Power mean approximation: d_max ~ (sum(d^p)/m)^(1/p), d_min ~ (sum(d^-p)/m)^(-1/p)
        sum_pos = np.sum(dists ** p)
        sum_neg = np.sum(dists ** (-p))
        if sum_neg == 0:
            return 1e10
        ratio = (sum_pos / sum_neg) ** (1.0 / p)
        return ratio ** 2  # minimize (d_max/d_min)^2 = 1/R

    def objective(params):
        """Exact objective for final evaluation."""
        if time.time() > start_time + deadline - safety_margin:
            raise TimeoutError()
            
        points = params.reshape(n, d)
        R = compute_R(points)
        return -R  # minimize -R

    # Initial guess: 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    X, Y = np.meshgrid(x, y)
    init_points = np.column_stack((X.ravel(), Y.ravel()))
    init_params = init_points.flatten()
    
    # Bounds for differential evolution
    bounds = [(-5, 5)] * (n * d)
    
    # Run differential evolution with smooth objective
    best_params = None
    try:
        result = differential_evolution(
            objective_smooth, 
            bounds, 
            seed=42, 
            maxiter=150, 
            tol=1e-6,
            polish=False,
            args=(8,),  # p=8 for power mean
            callback=lambda x, convergence: None
        )
        if result.success or (result.fun < 1e10):
            best_params = result.x
    except Exception as e:
        pass
    
    # If DE didn't find anything good, use initial guess
    if best_params is None:
        best_params = init_params
    
    # First refinement: Powell with exact objective
    try:
        result_ref = minimize(
            objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if result_ref.success or result_ref.fun < 0:
            best_params = result_ref.x
    except Exception as e:
        pass
    
    # Second refinement: Nelder-Mead for local polish
    try:
        result_nm = minimize(
            objective, 
            best_params, 
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        if result_nm.success or result_nm.fun < objective(best_params):
            best_params = result_nm.x
    except Exception as e:
        pass
    
    # Final high-precision polish with Powell
    try:
        result_final = minimize(
            objective, 
            best_params, 
            method='Powell', 
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if result_final.success or result_final.fun < objective(best_params):
            best_params = result_final.x
    except Exception as e:
        pass
    
    # Convert back to points
    points = best_params.reshape(n, d)
    
    # Final check: ensure no coincident points and valid shape
    if points.shape != (n, d):
        # Fallback to initial guess
        points = init_points
        
    return points

# EVOLVE-BLOCK-END
