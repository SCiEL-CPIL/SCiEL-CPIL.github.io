# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 550  # 550s limit, leaving 50s margin for safety
    
    n = 14
    d = 3
    
    def get_score(points):
        """Compute the metric (d_min/d_max)^2"""
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        """
        Negative of the score for minimization.
        Includes normalization to unit sphere and centering to remove invariances.
        """
        if time.time() > deadline:
            return 0.0 # Return a neutral value if timeout, though we should avoid this
            
        points = x.reshape(n, d)
        
        # Centering
        points = points - np.mean(points, axis=0)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1)
        # Handle zero norms
        norms[norms < 1e-10] = 1.0
        points = points / norms[:, np.newaxis]
        
        if not np.all(np.isfinite(points)):
            return 0.0
            
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            return 0.0
            
        # We want to maximize (d_min / d_max)^2, so minimize negative
        return -(d_min / d_max) ** 2

    def generate_icosahedron():
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    def generate_fibonacci_sphere(n_points):
        """Generate n points evenly distributed on a sphere."""
        points = []
        phi = (1 + np.sqrt(5)) / 2
        for i in range(n_points):
            z = 1 - (i + 0.5) * (2.0 / n_points)
            r = np.sqrt(1 - z**2)
            theta = phi * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    # Initial Configurations
    configs = []
    
    # 1. Icosahedron + 2 Poles
    ico = generate_icosahedron()
    poles = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
    config1 = np.vstack([ico, poles])
    # Normalize config1
    config1 = config1 - np.mean(config1, axis=0)
    config1 = config1 / np.linalg.norm(config1, axis=1)[:, np.newaxis]
    configs.append(config1)
    
    # 2. Icosahedron + 2 points at "equator" gaps
    # The icosahedron has 5-fold symmetry around z-axis. 
    # Let's try adding points at angles that might balance the structure.
    # A common perturbation is moving the poles slightly or adding points in the gaps.
    # Let's just use the Fibonacci sphere for a diverse start.
    fib = generate_fibonacci_sphere(14)
    fib = fib - np.mean(fib, axis=0)
    fib = fib / np.linalg.norm(fib, axis=1)[:, np.newaxis]
    configs.append(fib)
    
    # 3. Random
    np.random.seed(42)
    rand_pts = np.random.randn(14, 3)
    rand_pts = rand_pts - np.mean(rand_pts, axis=0)
    rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1)[:, np.newaxis]
    configs.append(rand_pts)
    
    best_points = None
    best_score = -np.inf
    
    # Strategy: 
    # 1. Try Nelder-Mead on each initial config.
    # 2. Try Differential Evolution for global search.
    # 3. Polish the best result from DE with Nelder-Mead.
    
    for idx, config in enumerate(configs):
        if time.time() > deadline - 10:
            break
            
        x0 = config.flatten()
        
        # Local Optimization with Nelder-Mead
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 20000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if res.fun < best_score:
                best_score = res.fun
                best_points = res.x.reshape(n, d)
                # Re-normalize and center for storage
                best_points = best_points - np.mean(best_points, axis=0)
                best_points = best_points / np.linalg.norm(best_points, axis=1)[:, np.newaxis]
        except Exception:
            pass
        
        # Check time
        if time.time() > deadline - 30:
            break

    # If we have time, run Differential Evolution for a more global search
    if time.time() < deadline - 60:
        # Bounds: Since we normalize, the scale doesn't matter much, but we need finite bounds.
        # Coordinates on unit sphere are in [-1, 1]. But the optimizer might drift.
        # Let's use bounds [-2, 2] to allow some drift before normalization.
        bounds = [(-2, 2)] * (n * d)
        
        try:
            # Use a reasonable population size and iterations
            # DE is stochastic, so set seed for reproducibility
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=123, 
                maxiter=150, 
                popsize=20, 
                tol=0.001,
                polish=True, # Let DE polish with its internal optimizer
                init='latinhypercube'
            )
            
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_points = res_de.x.reshape(n, d)
                best_points = best_points - np.mean(best_points, axis=0)
                best_points = best_points / np.linalg.norm(best_points, axis=1)[:, np.newaxis]
                
        except Exception:
            pass

    # Final Polish: Run NM on the best found so far with very high precision
    if best_points is not None and time.time() < deadline - 10:
        x_final = best_points.flatten()
        try:
            res_final = minimize(
                objective,
                x_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 50000,
                    'xatol': 1e-14,
                    'fatol': 1e-16,
                    'adaptive': True
                }
            )
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_points = res_final.x.reshape(n, d)
                best_points = best_points - np.mean(best_points, axis=0)
                best_points = best_points / np.linalg.norm(best_points, axis=1)[:, np.newaxis]
        except Exception:
            pass

    # Fallback if no optimization succeeded
    if best_points is None:
        best_points = configs[0]
        
    # Ensure the returned points are finite and valid
    if not np.all(np.isfinite(best_points)):
        best_points = configs[0]
        
    return best_points

# EVOLVE-BLOCK-END
