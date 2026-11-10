# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _evaluate(points: np.ndarray) -> float:
    """Compute the metric (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def _normalize(points: np.ndarray) -> np.ndarray:
    """Normalize points to lie on the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _soft_min(dists, beta):
    """
    Smooth approximation of the minimum using log-sum-exp.
    soft_min = -1/beta * log(sum(exp(-beta * d)))
    As beta -> inf, this approaches min(d).
    """
    # Shift for numerical stability
    min_d = np.min(dists)
    shifted = -beta * (dists - min_d)
    # Clip to avoid underflow/overflow in exp
    # exp(-700) is ~1e-304, exp(700) is ~1e304
    shifted = np.clip(shifted, -700, 700)
    sum_exp = np.sum(np.exp(shifted))
    return min_d - (1.0 / beta) * np.log(sum_exp)

def _objective_flat(x, beta):
    """
    Objective function for scipy.optimize.minimize.
    x: flat array of shape (14, 3)
    Returns: -soft_min(distances, beta)
    We minimize the negative soft min to maximize the minimum distance.
    """
    points = x.reshape(14, 3)
    # Normalize points to unit sphere inside the objective to enforce constraint
    # This is a "hard" constraint implementation via projection.
    # Note: Gradient of projection is not perfectly handled by BFGS if done naively,
    # but BFGS often works well with this "penalty/projection" style if steps are small.
    # Alternatively, we could parameterize via spherical coords, but that has singularities.
    # Let's stick to Cartesian + projection in the objective.
    # Actually, to be safe with gradients, we should ensure the points are normalized.
    # If we just normalize inside, the gradient w.r.t x is not simply the gradient of dists.
    # However, for BFGS, it approximates the Hessian.
    # A better approach: Use a penalty term for deviation from unit sphere, 
    # OR just normalize and accept that the optimizer sees a slightly distorted landscape.
    # Given the metric is scale invariant, fixing to unit sphere is correct.
    
    points = _normalize(points)
    dists = pdist(points)
    sm = _soft_min(dists, beta)
    return -sm

def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    return _normalize(vertices)

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Gradient-Based Optimization (BFGS) on a smooth surrogate of the minimum distance.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit
    
    # Initial configuration: Icosahedron + 2 poles
    ico = _icosahedron_vertices()
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    
    # We need 14 points. Icosahedron has 12.
    # Adding poles creates a configuration with high symmetry.
    initial_points = np.vstack([ico, p1.reshape(1, -1), p2.reshape(1, -1)])
    initial_points = _normalize(initial_points)
    
    best_points = initial_points.copy()
    best_metric = _evaluate(best_points)
    
    rng = np.random.RandomState(42)
    
    # Strategy:
    # 1. Start with a low beta (smooth objective, wide basins) to find a good basin.
    # 2. Increase beta (sharper objective) to refine the minimum distance.
    # 3. Use multiple restarts with random perturbations to escape local minima.
    
    # Betas to anneal through
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    current_points = best_points.copy()
    
    for beta in betas:
        # Do a few restarts at this beta level
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Perturb current best for restart
            if restart == 0:
                x0 = current_points.flatten()
            else:
                # Random restart from best found so far with noise
                noise = rng.randn(14, 3) * 0.1
                x0 = (current_points + noise).flatten()
                x0 = _normalize(x0.reshape(14, 3)).flatten()
            
            # Run BFGS
            # Max iterations to prevent hanging
            res = minimize(
                _objective_flat, 
                x0, 
                method='BFGS', 
                options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
            )
            
            if res.success or res.fun < _objective_flat(x0, beta):
                new_points = _normalize(res.x.reshape(14, 3))
                new_metric = _evaluate(new_points)
                
                if new_metric > best_metric:
                    best_metric = new_metric
                    best_points = new_points
                    current_points = new_points
    
    # Final refinement with a very high beta (close to true min)
    # Use Nelder-Mead or BFGS one last time on the best point
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        res = minimize(
            _objective_flat, 
            x0, 
            method='BFGS', 
            options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
        )
        if res.success:
            new_points = _normalize(res.x.reshape(14, 3))
            new_metric = _evaluate(new_points)
            if new_metric > best_metric:
                best_points = new_points
                best_metric = new_metric

    # Ensure finite
    if not np.all(np.isfinite(best_points)):
        best_points = initial_points
        
    return best_points

# EVOLVE-BLOCK-END
