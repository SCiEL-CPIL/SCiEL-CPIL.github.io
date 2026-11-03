# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses binary search on radius r and scipy.optimize.least_squares to find feasible configurations.
    """
    n = N
    
    # 1. Generate initial guess: Hexagonal lattice
    def generate_initial_guess(n_points):
        s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
        k = int(1.0 / s) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if x**2 + y**2 <= 0.9:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n_points:
            points = np.random.uniform(-0.9, 0.9, size=(n_points, 2))
        else:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)
            points = points[indices[:n_points]]
            
        return points

    x_init = generate_initial_guess(n)
    
    # 2. Define the residual function factory
    def make_residual_func(r_target, alpha=100.0):
        def residuals(x):
            centres = x.reshape(n, 2)
            dists = pdist(centres)
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            
            pair_violations = 2.0 * r_target - dists
            smooth_pair_viol = np.logaddexp(0.0, alpha * pair_violations) / alpha
            
            boundary_violations = r_target - boundary_dists
            smooth_boundary_viol = np.logaddexp(0.0, alpha * boundary_violations) / alpha
            
            return np.concatenate([smooth_pair_viol, smooth_boundary_viol])
        
        return residuals

    # 3. Binary search for the maximum r
    r_low = 0.04
    r_high = 0.06
    max_iterations = 25
    best_centres = x_init
    best_r = r_low
    current_x = x_init.flatten()
    
    # Helper to run least_squares
    def run_optimization(r_target, x_start):
        res_func = make_residual_func(r_target)
        try:
            result = least_squares(
                res_func, 
                x_start, 
                method='trf', 
                max_nfev=1000, 
                xtol=1e-10, 
                ftol=1e-10, 
                gtol=1e-10
            )
            cost = np.sum(result.fun**2)
            return result.x, cost
        except Exception:
            return x_start, np.inf

    for _ in range(max_iterations):
        r_mid = (r_low + r_high) / 2.0
        current_x, cost = run_optimization(r_mid, current_x)
        
        # If cost is small enough, consider it feasible
        if cost < 1e-6: 
            best_r = r_mid
            best_centres = current_x.reshape(n, 2)
            r_low = r_mid
        else:
            r_high = r_mid
            
        if r_high - r_low < 1e-6:
            break
            
    # Final run with best_r to ensure we have the best centres
    current_x, cost = run_optimization(best_r, current_x)
    if cost < 1e-5:
        best_centres = current_x.reshape(n, 2)
        
    if not np.all(np.isfinite(best_centres)):
        return generate_initial_guess(n)
        
    return best_centres

# EVOLVE-BLOCK-END
