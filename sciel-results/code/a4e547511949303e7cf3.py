# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses gradient-based optimization with multiple restarts on a unit sphere.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    deadline = time.time() + 580  # Safety margin before 630s limit
    
    def objective(params):
        """Negative log of (d_min/d_max)^2 to minimize"""
        pts = params.reshape(n, d)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 0 or d_max <= 0:
            return 1e10
        ratio_sq = (d_min / d_max) ** 2
        return -np.log(ratio_sq)
    
    def objective_grad(params):
        """Gradient of the objective function"""
        pts = params.reshape(n, d)
        # Compute all pairwise distances
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_min <= 0 or d_max <= 0:
            return np.zeros_like(params)
        
        ratio_sq = (d_min / d_max) ** 2
        log_ratio = np.log(ratio_sq)
        
        # We want to maximize log_ratio, so minimize -log_ratio
        # d/dx (-log_ratio) = -1/ratio_sq * dratio_sq/dx
        # ratio_sq = (d_min/d_max)^2
        # d(ratio_sq)/dx = 2*d_min/d_max^2 * (dd_min/dx * d_max - d_min * dd_max/dx) / d_max
        # = 2*d_min/d_max^2 * (dd_min/dx - d_min * dd_max/dx / d_max)
        
        # Actually let's compute numerically for stability
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_p = params.copy()
            params_p[i] += eps
            f_p = objective(params_p)
            params_m = params.copy()
            params_m[i] -= eps
            f_m = objective(params_m)
            grad[i] = (f_p - f_m) / (2 * eps)
        
        return grad
    
    def fibonacci_sphere(n_points, radius=1.0):
        """Generate points on a sphere using Fibonacci lattice"""
        points = np.zeros((n_points, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_points):
            z = 1 - (2 * i + 1) / n_points
            r = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / golden_ratio
            points[i] = radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
        return points
    
    def random_sphere(n_points, radius=1.0, seed=None):
        """Generate random points on a sphere"""
        if seed is not None:
            np.random.seed(seed)
        points = np.random.randn(n_points, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = radius * points / norms
        return points
    
    def optimize_from_initial(pts0, max_iter=5000):
        """Run L-BFGS-B optimization from initial point"""
        params0 = pts0.flatten()
        result = minimize(
            objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x.reshape(n, d), result.fun
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initializations
    initializations = []
    
    # Fibonacci sphere
    initializations.append(fibonacci_sphere(n))
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1011, 2020, 3030, 4040]:
        initializations.append(random_sphere(n, seed=seed))
    
    # Slightly perturbed Fibonacci
    fib = fibonacci_sphere(n)
    for perturb_seed in [1, 2, 3]:
        np.random.seed(perturb_seed)
        perturbed = fib + np.random.randn(n, d) * 0.1
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        initializations.append(perturbed / norms)
    
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        try:
            opt_pts, opt_val = optimize_from_initial(init_pts, max_iter=3000)
            
            # Compute actual score
            dists = pdist(opt_pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_min > 0 and d_max > 0:
                score = (d_min / d_max) ** 2
                if score > best_score:
                    best_score = score
                    best_points = opt_pts
        except Exception:
            continue
    
    # If we still have time, try more refinements on the best solution
    if best_points is not None and time.time() < deadline - 30:
        try:
            # Try Nelder-Mead refinement
            nm_result = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            nm_pts = nm_result.x.reshape(n, d)
            dists = pdist(nm_pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_min > 0 and d_max > 0:
                score = (d_min / d_max) ** 2
                if score > best_score:
                    best_score = score
                    best_points = nm_pts
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: return Fibonacci sphere
        best_points = fibonacci_sphere(n)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
