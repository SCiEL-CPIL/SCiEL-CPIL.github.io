# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _compute_score(points):
    """Compute the metric: (d_min / d_max)^2"""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective_and_grad(x, n, mu):
    """
    Objective: minimize -log(d_min_sq) - mu * sum(log(1 - d_ij_sq))
    where d_ij_sq are squared pairwise distances, constrained to be < 1.
    """
    points = x.reshape(n, 3)
    
    # Compute all pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d_sq = np.sum(diff ** 2, axis=2)
    
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    d_sq_upper = d_sq[iu]
    
    # Check if any distance >= 1 (constraint violation)
    if np.any(d_sq_upper >= 1.0):
        # Return a large penalty
        penalty = np.sum(np.maximum(0, d_sq_upper - 1.0)) ** 2
        return 1e6 + penalty, np.zeros_like(x)
    
    d_min_sq = np.min(d_sq_upper)
    
    # Log barrier term: -mu * sum(log(1 - d_sq))
    log_barrier = -mu * np.sum(np.log(1.0 - d_sq_upper))
    
    # Objective: -log(d_min_sq) + log_barrier
    # We want to maximize d_min_sq, so minimize -log(d_min_sq)
    obj = -np.log(d_min_sq) + log_barrier
    
    # Gradient with respect to x
    # d/dx of -log(d_min_sq) where d_min_sq = ||p_i - p_j||^2 for the min pair
    # d/dx of log_barrier = -mu * sum over pairs of d/dx log(1 - d_ij_sq)
    
    grad = np.zeros_like(x)
    
    # Find the minimum pair
    min_idx = np.argmin(d_sq_upper)
    
    # Get the pair indices from upper triangle
    iu_i, iu_j = iu
    min_i = iu_i[min_idx]
    min_j = iu_j[min_idx]
    
    # Gradient of -log(d_min_sq) w.r.t. points
    # d_min_sq = ||p_i - p_j||^2
    # d/dp_i = -2*(p_i - p_j) / d_min_sq
    # d/dp_j = 2*(p_i - p_j) / d_min_sq
    diff_min = points[min_i] - points[min_j]
    grad[min_i] += -2.0 * diff_min / d_min_sq
    grad[min_j] += 2.0 * diff_min / d_min_sq
    
    # Gradient of log barrier
    # d/dx of -mu * sum(log(1 - d_k_sq)) = -mu * sum(1/(1-d_k_sq)) * d/dx(1-d_k_sq)
    # = -mu * sum(1/(1-d_k_sq)) * (-d/dx(d_k_sq))
    # = mu * sum(1/(1-d_k_sq)) * d/dx(d_k_sq)
    
    # For each pair (a,b) with d_k_sq = ||p_a - p_b||^2
    # d/dp_a = 2*(p_a - p_b), d/dp_b = -2*(p_a - p_b)
    
    for k in range(len(iu_i)):
        a = iu_i[k]
        b = iu_j[k]
        dk_sq = d_sq_upper[k]
        factor = mu / (1.0 - dk_sq)
        diff_k = points[a] - points[b]
        grad[a] += 2.0 * factor * diff_k
        grad[b] += -2.0 * factor * diff_k
    
    return obj, grad


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    Uses BFGS optimization with log-barrier constraint on maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 600  # Safety margin: 600s out of 630s
    
    n = 14
    d = 3
    
    best_points = None
    best_score = -1.0
    
    # Try multiple random restarts
    np.random.seed(42)
    
    n_restarts = 50  # Will be limited by time
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:  # Leave 30s margin
            break
        
        # Initialize points: random on unit sphere or in unit ball
        if restart % 2 == 0:
            # On unit sphere
            points = np.random.randn(n, d)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        else:
            # In unit ball
            points = np.random.randn(n, d)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms * np.random.uniform(0.3, 1.0, size=(n, 1))
        
        x0 = points.flatten()
        
        # Start with small mu and increase
        mu = 0.1
        current_x = x0.copy()
        
        # Iteratively increase mu (barrier parameter)
        for mu_iter in range(8):
            if time.time() > deadline - 10:
                break
            
            # Scale mu
            mu_val = mu * (10 ** mu_iter)
            
            try:
                result = minimize(
                    _objective_and_grad,
                    current_x,
                    args=(n, mu_val),
                    method='BFGS',
                    jac=True,
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < 1e5:
                    current_x = result.x
                else:
                    # If optimization failed, keep previous
                    pass
                    
            except Exception:
                break
        
        # Evaluate the result
        final_points = current_x.reshape(n, 3)
        
        # Check validity
        if not np.all(np.isfinite(final_points)):
            continue
        
        score = _compute_score(final_points)
        
        if score > best_score:
            best_score = score
            best_points = final_points.copy()
        
        # If we've found a very good solution, we can stop early
        if best_score > 6.0:
            break
    
    if best_points is None:
        # Fallback: random points
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points


# EVOLVE-BLOCK-END
