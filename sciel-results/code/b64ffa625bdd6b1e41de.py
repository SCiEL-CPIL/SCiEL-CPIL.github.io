# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a non-convex quadratic programming approach with BFGS optimization
    to maximize the minimum pairwise squared distance subject to the maximum
    pairwise squared distance being fixed to 1.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute pair indices for efficiency
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
    pair_indices = np.array(pair_indices)
    n_pairs = len(pair_indices)
    
    # Objective: maximize min pairwise squared distance
    # We minimize the negative of min pairwise squared distance
    # Subject to max pairwise squared distance <= 1 (using barrier)
    
    def objective_and_grad(x):
        """
        x is flattened array of shape (n*d,)
        Returns objective value and gradient
        """
        points = x.reshape(n, d)
        
        # Compute all pairwise squared distances
        # Using broadcasting: (n, 1, d) - (1, n, d)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        
        # Extract pairwise distances (upper triangle)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_sq_dists = sq_dists[i_idx, j_idx]
        
        # Find min and max
        min_sq_dist = np.min(pair_sq_dists)
        max_sq_dist = np.max(pair_sq_dists)
        
        # Barrier term for max constraint: -log(1 - max_sq_dist) if max_sq_dist < 1
        # If max_sq_dist >= 1, add large penalty
        if max_sq_dist >= 1.0:
            # Large penalty
            barrier = 1e6 * (max_sq_dist - 1.0) ** 2
        else:
            barrier = -np.log(1.0 - max_sq_dist + 1e-12)
        
        # Objective: -min_sq_dist + barrier
        obj = -min_sq_dist + barrier
        
        # Gradient computation
        # d(-min_sq_dist)/dx: only the pair(s) achieving min contribute
        min_idx = np.argmin(pair_sq_dists)
        min_i, min_j = pair_indices[min_idx]
        
        # d(min_sq_dist)/dx_k = 2 * (point_i - point_j)_k for the min pair
        # So d(-min_sq_dist)/dx = -2 * (point_i - point_j)
        grad = np.zeros_like(x)
        diff_min = points[min_i] - points[min_j]
        grad[min_i * d:(min_i + 1) * d] -= 2.0 * diff_min
        grad[min_j * d:(min_j + 1) * d] += 2.0 * diff_min
        
        # Gradient of barrier term
        if max_sq_dist < 1.0:
            max_idx = np.argmax(pair_sq_dists)
            max_i, max_j = pair_indices[max_idx]
            # d(-log(1 - max_sq_dist))/dx = 1/(1 - max_sq_dist) * d(max_sq_dist)/dx
            # d(max_sq_dist)/dx_k = 2 * (point_max_i - point_max_j)_k
            diff_max = points[max_i] - points[max_j]
            barrier_grad_factor = 1.0 / (1.0 - max_sq_dist + 1e-12)
            grad[max_i * d:(max_i + 1) * d] += 2.0 * barrier_grad_factor * diff_max
            grad[max_j * d:(max_j + 1) * d] -= 2.0 * barrier_grad_factor * diff_max
        else:
            # Gradient of penalty term
            max_idx = np.argmax(pair_sq_dists)
            max_i, max_j = pair_indices[max_idx]
            diff_max = points[max_i] - points[max_j]
            penalty_grad_factor = 2.0 * 1e6 * (max_sq_dist - 1.0)
            grad[max_i * d:(max_i + 1) * d] += penalty_grad_factor * diff_max
            grad[max_j * d:(max_j + 1) * d] -= penalty_grad_factor * diff_max
        
        return obj, grad
    
    # Initial guess: use a reasonable starting configuration
    # Start with points on a sphere or random
    np.random.seed(42)
    x0 = np.random.randn(n * d) * 0.5
    
    # Run BFGS optimization
    result = minimize(
        objective_and_grad,
        x0,
        method='BFGS',
        jac=True,
        options={'maxiter': 500, 'gtol': 1e-8, 'disp': False}
    )
    
    x_opt = result.x
    points = x_opt.reshape(n, d)
    
    # Normalize so that max pairwise distance is 1 (to satisfy constraint)
    # Actually, we want to maximize min/max ratio, so we should scale
    # Let's recompute and ensure the points are valid
    
    # Verify the result
    if not np.all(np.isfinite(points)):
        # Fallback to random
        np.random.seed(42)
        points = np.random.randn(n, d)
    
    return points


# EVOLVE-BLOCK-END
