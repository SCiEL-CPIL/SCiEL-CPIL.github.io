import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.special import logsumexp
import time

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600
    
    def score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2
    
    def fibonacci_sphere(n_points, radius=1.0):
        """Generate points on a sphere using Fibonacci spiral."""
        if n_points == 1:
            return np.array([[0, 0, radius]])
        points = np.zeros((n_points, 3))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            r = np.sqrt(1 - y*y)
            theta = golden_angle * i
            points[i] = [np.cos(theta) * r * radius, y * radius, np.sin(theta) * r * radius]
        return points
    
    def soft_min(dists, beta):
        """Numerically stable soft minimum using logsumexp."""
        neg_dists = -dists
        return -logsumexp(beta * neg_dists) / beta
    
    def soft_max(dists, beta):
        """Numerically stable soft maximum using logsumexp."""
        return logsumexp(beta * dists) / beta
    
    def smooth_objective(x, beta):
        """Smoothed objective using log-sum-exp for soft min/max."""
        p = x.reshape(n, d)
        if not np.all(np.isfinite(p)):
            return 1.0
        dists = pdist(p)
        if np.max(dists) < 1e-12:
            return 1.0
        d_min_soft = soft_min(dists, beta)
        d_max_soft = soft_max(dists, beta)
        if d_max_soft < 1e-12:
            return 1.0
        ratio = d_min_soft / d_max_soft
        return -(ratio ** 2)
    
    # Start with Fibonacci sphere points
    best_points = fibonacci_sphere(n)
    best_score = score(best_points)
    
    # Beta annealing schedule
    beta_schedule = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in beta_schedule:
        if time.time() - start_time > time_limit - 30:
            break
        
        def obj_annealed(x):
            return smooth_objective(x, beta)
        
        try:
            res = minimize(
                obj_annealed,
                best_points.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            new_points = res.x.reshape(n, d)
            new_score = score(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
        except Exception:
            continue
    
    # Final polish with exact objective
    if time.time() - start_time < time_limit - 10:
        def obj_polish(x):
            p = x.reshape(n, d)
            s = score(p)
            return -s
        
        try:
            res_polish = minimize(
                obj_polish,
                best_points.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res_polish.fun > best_score:
                best_points = res_polish.x.reshape(n, d)
                best_score = -res_polish.fun
        except Exception:
            pass
    
    # Final validity check
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    return best_points
