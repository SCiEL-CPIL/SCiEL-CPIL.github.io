# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    timeout = 600.0
    safety_margin = 10.0
    
    def angles_to_points(angles):
        # angles is shape (n, 2): theta (polar, 0 to pi), phi (azimuth, 0 to 2*pi)
        # Ensure angles are within bounds to avoid numerical issues, though NM handles it
        theta = angles[:, 0]
        phi = angles[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))

    def objective(angles):
        points = angles_to_points(angles)
        # pdist returns condensed form of pairwise distances
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 0.0
            
        ratio_sq = (d_min / d_max) ** 2
        return -ratio_sq

    def fibonacci_sphere(n):
        """Generate n points on a sphere using Fibonacci lattice."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            # i+0.5 to avoid poles exactly
            theta = np.arccos(1 - 2 * (i + 0.5) / n)
            phi_angle = 2 * np.pi * i / phi
            points.append([theta, phi_angle])
        return np.array(points)

    # Multi-start Nelder-Mead
    best_score = -np.inf
    best_points = None
    
    # We will try several random restarts and the Fibonacci initialization
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - safety_margin:
            break
            
        if restart == 0:
            # Start with a structured, well-distributed initialization
            init_angles = fibonacci_sphere(n)
        else:
            # Random initialization
            np.random.seed(42 + restart)
            init_angles = np.random.rand(n, 2)
            init_angles[:, 0] *= np.pi
            init_angles[:, 1] *= 2 * np.pi
            
        try:
            res = minimize(objective, init_angles, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = angles_to_points(res.x)
        except Exception:
            continue

    # If optimization failed or didn't improve, fall back to Fibonacci sphere points
    if best_points is None:
        init_angles = fibonacci_sphere(n)
        best_points = angles_to_points(init_angles)
        # Calculate score for fallback just in case, though we just return points
        dists = pdist(best_points)
        if len(dists) > 0:
             d_min = np.min(dists)
             d_max = np.max(dists)
             if d_max > 1e-12:
                 best_score = (d_min / d_max) ** 2

    return best_points

# EVOLVE-BLOCK-END
