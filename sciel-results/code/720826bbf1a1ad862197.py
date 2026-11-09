# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def _compute_scores(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max < 1e-12:
        return 0.0
    return (d_min / d_max) ** 2


def _smooth_objective(params, n, d, t):
    """
    Smooth objective using log-sum-exp for min and max distances.
    We maximize log(d_min) - log(d_max), which is equivalent to maximizing d_min/d_max.
    Using soft-min (log-sum-exp with negative) and soft-max (log-sum-exp).
    
    params: flattened array of shape (n*d,)
    Returns: negative of the objective (for minimization)
    """
    pts = params.reshape(n, d)
    # Compute pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Use log-sum-exp for stability
    # soft-max of distances: log(sum(exp(d/t)))
    # soft-min of distances: -log(sum(exp(-d/t)))
    # But we work with squared distances for efficiency, then take sqrt
    
    # Actually, let's work with distances directly for the ratio
    dists = np.sqrt(np.maximum(sq_dists, 0))
    np.fill_diagonal(dists, np.inf)
    
    # Soft max: log-sum-exp(dists / t)
    max_dists = np.max(dists, axis=1)
    log_sum_exp_max = np.log(np.sum(np.exp((dists - max_dists[:, np.newaxis]) / t), axis=1) * np.exp(max_dists / t))
    soft_max = np.mean(log_sum_exp_max)  # Average over all points
    
    # Soft min: -log-sum-exp(-dists / t)
    min_dists = np.min(dists, axis=1)
    log_sum_exp_min_neg = np.log(np.sum(np.exp((-dists - min_dists[:, np.newaxis]) / t), axis=1) * np.exp(-min_dists / t))
    soft_min = -np.mean(log_sum_exp_min_neg)  # Average over all points
    
    # Objective: maximize soft_min - soft_max (i.e., minimize negative)
    return -(soft_min - soft_max)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Initial configuration: regular 16-gon on a circle
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Also try a few other initializations
    inits = [
        initial_points,
        # Two concentric circles
        np.vstack([
            np.column_stack([np.cos(angles[:8]), np.sin(angles[:8])]),
            np.column_stack([0.5 * np.cos(angles[8:]), 0.5 * np.sin(angles[8:])])
        ]),
        # Random perturbation of circle
        initial_points + 0.1 * np.random.randn(n, d),
    ]
    
    best_points = initial_points
    best_score = _compute_scores(initial_points)
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        params0 = init.flatten()
        
        # Try different temperature values for the smooth surrogate
        temperatures = [0.01, 0.05, 0.1, 0.5]
        
        for t in temperatures:
            if time.time() > deadline - 30:
                break
            
            try:
                result = minimize(
                    _smooth_objective,
                    params0,
                    args=(n, d, t),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 0:
                    candidate = result.x.reshape(n, d)
                    score = _compute_scores(candidate)
                    
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
                        
            except Exception:
                continue
    
    # Final polish: try to improve with a tighter optimization
    if time.time() < deadline - 10:
        try:
            # Use the best points found so far as starting point
            params0 = best_points.flatten()
            
            # Try with smaller temperature for more precise optimization
            for t in [0.001, 0.005, 0.01]:
                if time.time() > deadline - 10:
                    break
                
                result = minimize(
                    _smooth_objective,
                    params0,
                    args=(n, d, t),
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success:
                    candidate = result.x.reshape(n, d)
                    score = _compute_scores(candidate)
                    
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
                        params0 = best_points.flatten()
                        
        except Exception:
            pass
    
    return best_points


# EVOLVE-BLOCK-END
