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
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def _negative_metric(points_flat: np.ndarray) -> float:
    """Objective function for minimization: negative of the metric."""
    n = len(points_flat) // 3
    points = points_flat.reshape((n, 3))
    # Normalize points to unit sphere to constrain the search space?
    # The problem does not require points to be on a sphere, but the ratio is scale-invariant.
    # However, unconstrained optimization might drift.
    # Let's normalize to unit sphere inside the objective to keep points bounded.
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    score = _evaluate(points)
    return -score


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n points on a unit sphere using Fibonacci lattice."""
    points = np.zeros((n, 3))
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - (2 * i + 1) / n
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.

    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    n = 14
    d = 3
    
    best_points = _fibonacci_sphere(n)
    best_score = _evaluate(best_points)
    
    # 1. Try Nelder-Mead from Fibonacci start
    x0 = best_points.flatten()
    try:
        res = minimize(_negative_metric, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success or res.fun < 0:
            cand_points = res.x.reshape((n, d))
            # Normalize
            norms = np.linalg.norm(cand_points, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            cand_points = cand_points / norms
            score = _evaluate(cand_points)
            if score > best_score:
                best_points = cand_points
                best_score = score
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_points

    # 2. Try multiple random restarts with Nelder-Mead
    np.random.seed(42)
    for i in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization on sphere
        init = np.random.randn(n, d)
        init = init / np.linalg.norm(init, axis=1, keepdims=True)
        
        x0 = init.flatten()
        try:
            # Use a tighter tolerance for speed in later iterations
            res = minimize(_negative_metric, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success or res.fun < 0:
                cand_points = res.x.reshape((n, d))
                norms = np.linalg.norm(cand_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                cand_points = cand_points / norms
                score = _evaluate(cand_points)
                if score > best_score:
                    best_points = cand_points
                    best_score = score
        except Exception:
            pass
            
    # 3. Final polish with a very high precision Nelder-Mead
    if time.time() < deadline - 60:
        x0 = best_points.flatten()
        try:
            res = minimize(_negative_metric, x0, method='Nelder-Mead', 
                           options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success or res.fun < 0:
                cand_points = res.x.reshape((n, d))
                norms = np.linalg.norm(cand_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                cand_points = cand_points / norms
                score = _evaluate(cand_points)
                if score > best_score:
                    best_points = cand_points
        except Exception:
            pass

    # Ensure points are finite
    if not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(n)
    
    return best_points


# EVOLVE-BLOCK-END
