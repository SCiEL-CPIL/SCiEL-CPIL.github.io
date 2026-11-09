import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Finds 14 points in 3D space that maximize the ratio of the minimum 
    pairwise distance to the maximum pairwise distance.
    
    Returns:
        np.ndarray: Array of shape (14, 3) containing the coordinates.
    """
    start_time = time.time()
    deadline = 580.0  # Safety margin before 600s kill
    
    n_points = 14
    dim = 3

    def compute_metric(points):
        """Compute the actual metric (d_min / d_max)^2"""
        if points.shape != (n_points, dim):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        d = pdist(points)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        return (d_min / d_max) ** 2

    def objective_soft(points_flat, beta):
        """
        Objective function to minimize: negative log ratio of soft-min to soft-max.
        points_flat: 1D array of size 42
        beta: temperature parameter for soft-min/max
        """
        points = points_flat.reshape(n_points, dim)
        
        # Ensure numerical stability: if points are huge or zero, pdist might have issues
        # But we initialize on unit sphere, so scale should be reasonable.
        
        d = pdist(points)
        
        # Handle case where distances are 0 (coincident points)
        # pdist returns 0 for coincident points.
        # logsumexp(-beta * d) will be dominated by large beta*0 = 0 if d=0?
        # exp(-beta * 0) = 1.
        # If d has zeros, soft_min will be close to 0.
        
        log_sum_min = logsumexp(-beta * d)
        log_sum_max = logsumexp(beta * d)
        
        soft_min = -1.0 / beta * log_sum_min
        soft_max =  1.0 / beta * log_sum_max
        
        # Avoid log(0)
        if soft_min <= 1e-10 or soft_max <= 1e-10:
            return 1e10
            
        # We want to maximize log(soft_min) - log(soft_max)
        # So we minimize the negative of that
        return -(np.log(soft_min) - np.log(soft_max))

    def fibonacci_sphere(n):
        """
        Generate n points on a unit sphere using the Fibonacci sphere algorithm.
        This provides a good initial distribution for Tammes-like problems.
        """
        points = []
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        pts = np.column_stack((x, y, z))
        return pts

    best_score = 0.0
    best_points = None
    
    # Phase 1: Optimization from Fibonacci Sphere Start
    # This is the most promising start.
    
    # Initialize with Fibonacci sphere
    p_fib = fibonacci_sphere(n_points)
    
    # Run optimization with multiple betas to ensure we find the sharp minimum
    betas = [10, 50, 200, 1000]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
            
        x0 = p_fib.flatten()
        
        try:
            res = minimize(
                objective_soft, 
                x0, 
                args=(beta,),
                method='Powell', 
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if res.success or res.fun < 1e9: # Even if not "success", check score
                p_opt = res.x.reshape(n_points, dim)
                # Normalize to unit diameter to prevent drift, although objective is scale invariant
                # Normalizing helps if the optimizer drifts in scale.
                d_opt = pdist(p_opt)
                d_max_opt = np.max(d_opt)
                if d_max_opt > 0:
                    p_opt = p_opt / d_max_opt
                
                score = compute_metric(p_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = p_opt
                    # If we find a very good score, we might not need further random starts
                    # But let's allow a bit more search.
                    
        except Exception:
            continue

    # Phase 2: If time remains, try a few random starts with Fibonacci perturbation or just random
    # to ensure we didn't get stuck in a local minimum specific to the Fibonacci start.
    # However, Fibonacci start is very robust.
    
    if time.time() < deadline - 60:
        n_random_starts = 10
        for i in range(n_random_starts):
            if time.time() > deadline - 30:
                break
            
            # Random start on sphere
            np.random.seed(42 + i)
            vecs = np.random.randn(n_points, dim)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            p_rand = vecs / norms
            
            x0 = p_rand.flatten()
            
            # Use a moderate beta for random starts
            try:
                res = minimize(
                    objective_soft, 
                    x0, 
                    args=(50,),
                    method='Powell', 
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if res.success or res.fun < 1e9:
                    p_opt = res.x.reshape(n_points, dim)
                    d_opt = pdist(p_opt)
                    d_max_opt = np.max(d_opt)
                    if d_max_opt > 0:
                        p_opt = p_opt / d_max_opt
                    
                    score = compute_metric(p_opt)
                    
                    if score > best_score:
                        best_score = score
                        best_points = p_opt
                        
            except Exception:
                continue

    if best_points is None:
        # Fallback: Return the Fibonacci sphere points (which are valid)
        best_points = fibonacci_sphere(n_points)
        
    return best_points
