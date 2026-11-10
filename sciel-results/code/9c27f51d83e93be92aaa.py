# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def _compute_score(points: np.ndarray) -> float:
    """Compute R = (d_min / d_max)^2 for the given points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _objective_and_grad(x: np.ndarray) -> tuple:
    """
    Objective: minimize sum_{i<j} (d_ij^2 - mu)^2 where mu is adaptive.
    We use mu = mean of squared distances to keep spread controlled.
    This is smooth and differentiable.
    """
    n = x.shape[0]
    pts = x.reshape(n, 2)
    
    # Compute all pairwise squared distances
    # d_ij^2 = |p_i - p_j|^2
    # Use matrix form: D2[i,j] = |p_i - p_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    
    # Only upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    d2_upper = d2[iu]
    
    mu = np.mean(d2_upper)
    
    # Objective: sum (d2 - mu)^2
    obj = np.sum((d2_upper - mu) ** 2)
    
    # Gradient with respect to x
    # d/dx_k of (d2_ij - mu)^2 = 2(d2_ij - mu) * d/dx_k (d2_ij - mu)
    # d/dx_k (d2_ij) = 2*(p_i - p_j)_k for the relevant coordinate
    # d/dx_k (mu) = (1/m) * sum_{a<b} d/dx_k (d2_ab)
    
    m = len(d2_upper)
    grad = np.zeros_like(x)
    
    # For each pair (i,j), contribution to gradient:
    # 2*(d2_ij - mu) * [2*(p_i - p_j) - (1/m)*sum_{a<b} 2*(p_a - p_b)]
    # The second term is the gradient of mu w.r.t. x
    
    # Let's compute gradient of mu:
    # mu = (1/m) * sum_{a<b} d2_ab
    # d mu / dx_k = (1/m) * sum_{a<b} 2*(p_a - p_b)_k
    
    # We can compute this by accumulating
    grad_mu = np.zeros_like(x)
    for idx in range(m):
        i, j = iu[0][idx], iu[1][idx]
        # d2_ij = |p_i - p_j|^2
        # d/dx of d2_ij: for coordinate k of point i: 2*(p_i[k] - p_j[k])
        # for coordinate k of point j: 2*(p_j[k] - p_i[k])
        diff_ij = pts[i] - pts[j]
        grad_mu[i * 2:(i + 1) * 2] += 2 * diff_ij
        grad_mu[j * 2:(j + 1) * 2] += -2 * diff_ij
    
    grad_mu /= m
    
    # Now compute full gradient
    for idx in range(m):
        i, j = iu[0][idx], iu[1][idx]
        diff_ij = pts[i] - pts[j]
        d2_ij = np.sum(diff_ij ** 2)
        coeff = 2 * (d2_ij - mu)
        # d/dx of (d2_ij - mu)^2 = 2*(d2_ij - mu) * (d/dx d2_ij - d/dx mu)
        # d/dx d2_ij for point i: 2*diff_ij, for point j: -2*diff_ij
        grad[i * 2:(i + 1) * 2] += coeff * (2 * diff_ij - grad_mu[i * 2:(i + 1) * 2])
        grad[j * 2:(j + 1) * 2] += coeff * (-2 * diff_ij - grad_mu[j * 2:(j + 1) * 2])
    
    return obj, grad


def _local_optimize(x0: np.ndarray, max_iter: int = 5000, tol: float = 1e-10) -> tuple:
    """
    Perform gradient descent with adaptive step size on the smooth objective.
    Returns (final_points, final_score).
    """
    n = x0.shape[0] // 2
    x = x0.copy()
    best_x = x.copy()
    best_score = _compute_score(x.reshape(n, 2))
    
    lr = 0.01
    prev_obj = np.inf
    
    for it in range(max_iter):
        obj, grad = _objective_and_grad(x)
        
        # Check convergence
        if abs(prev_obj - obj) < tol and it > 100:
            break
        prev_obj = obj
        
        # Adaptive step size
        x_new = x - lr * grad
        
        # Compute score of new point
        score_new = _compute_score(x_new.reshape(n, 2))
        score_cur = _compute_score(x.reshape(n, 2))
        
        if score_new > score_cur:
            x = x_new
            if score_new > best_score:
                best_score = score_new
                best_x = x.copy()
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.8, 1e-6)
    
    return best_x, best_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    deadline = time.time() + 600  # 600 second limit with safety margin
    
    # Multiple initializations and local optimizations
    best_points = None
    best_score = 0.0
    
    # Strategy 1: Regular polygon initialization
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    pts, score = _local_optimize(x0, max_iter=3000)
    if score > best_score:
        best_score = score
        best_points = pts.reshape(n, 2)
    
    if time.time() > deadline - 120:
        return best_points
    
    # Strategy 2: Grid-like initialization
    # Place points on a 4x4 grid, then perturb
    grid_x = np.array([0, 1, 2, 3])
    grid_y = np.array([0, 1, 2, 3])
    gx, gy = np.meshgrid(grid_x, grid_y)
    x0 = np.column_stack([gx.flatten(), gy.flatten()]).flatten()
    pts, score = _local_optimize(x0, max_iter=3000)
    if score > best_score:
        best_score = score
        best_points = pts.reshape(n, 2)
    
    if time.time() > deadline - 120:
        return best_points
    
    # Strategy 3: Two concentric circles
    n_outer = 8
    n_inner = 8
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_outer
    outer = np.column_stack([2 * np.cos(angles_outer), 2 * np.sin(angles_outer)])
    inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    x0 = np.vstack([outer, inner]).flatten()
    pts, score = _local_optimize(x0, max_iter=3000)
    if score > best_score:
        best_score = score
        best_points = pts.reshape(n, 2)
    
    if time.time() > deadline - 120:
        return best_points
    
    # Strategy 4: Random initialization with multiple restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 60:
            break
        x0 = np.random.randn(n * d) * 2
        pts, score = _local_optimize(x0, max_iter=2000)
        if score > best_score:
            best_score = score
            best_points = pts.reshape(n, 2)
    
    # Final refinement with more iterations
    if time.time() < deadline - 30 and best_points is not None:
        x0 = best_points.flatten()
        pts, score = _local_optimize(x0, max_iter=10000)
        if score > best_score:
            best_score = score
            best_points = pts.reshape(n, 2)
    
    return best_points


# EVOLVE-BLOCK-END
