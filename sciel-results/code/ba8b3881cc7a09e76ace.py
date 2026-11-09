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


def _repulsion_optimize(points: np.ndarray, n_iter: int = 2000, lr: float = 0.01) -> np.ndarray:
    """
    Optimize points on the unit sphere to maximize minimum distance using
    repulsion forces (Tammes problem approach).
    """
    points = points.copy()
    n = len(points)
    
    for it in range(n_iter):
        # Compute repulsion forces
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random nudge if points coincide
                    diff = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2 (Coulomb-like)
                force_mag = 1.0 / (dist * dist)
                force_vec = diff / dist * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Update positions
        points += lr * forces
        
        # Project back to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.99
    
    return points


def _log_sum_exp_surrogate(points: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
    """
    Compute a smooth surrogate for the negative log-sum-exp of pairwise distances.
    This is a smooth approximation to -log(d_min) which is equivalent to minimizing
    a smooth function that penalizes small distances.
    
    Returns:
        value: The surrogate value (lower is better, so we minimize this)
        grad: The gradient with respect to the flattened points array
    """
    n = len(points)
    pts = points.reshape(-1)  # Flatten for optimization
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Log-sum-exp of negative distances
    # We want to maximize min(dists), which is equivalent to minimizing -log(sum(exp(-beta * dists)))
    # For large beta, this approaches -log(d_min)
    
    neg_beta_dists = -beta * dists
    # Subtract max for numerical stability
    max_val = np.max(neg_beta_dists)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_dists - max_val)))
    
    value = log_sum
    
    # Gradient
    # d/dx_i log(sum_j exp(-beta * d_ij)) = sum_j [exp(-beta * d_ij) * (-beta * d/dx_i d_ij)] / sum_j exp(-beta * d_ij)
    # d/dx_i d_ij = (x_i - x_j) / d_ij
    
    exp_vals = np.exp(neg_beta_dists - max_val)
    sum_exp = np.sum(exp_vals)
    
    grad = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff_ij = points[i] - points[j]
            dist_ij = dists[i, j]
            if dist_ij < 1e-10:
                continue
            # Gradient contribution
            coeff = exp_vals[i, j] / sum_exp * (-beta) * (-1.0 / dist_ij)
            grad[i * 3:(i + 1) * 3] += coeff * diff_ij
    
    return value, grad


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    n = 14
    d = 3
    
    # Start with Fibonacci sphere initialization
    best_points = _fibonacci_sphere(n)
    best_score = _evaluate(best_points)
    
    # Phase 1: Repulsion optimization on unit sphere
    points = _repulsion_optimize(best_points.copy(), n_iter=3000, lr=0.005)
    score = _evaluate(points)
    if score > best_score:
        best_points = points.copy()
        best_score = score
    
    # Check time
    if time.time() > deadline - 100:
        return best_points
    
    # Phase 2: L-BFGS-B optimization with log-sum-exp surrogate
    # We optimize in the space of point coordinates, but constrain to unit sphere
    # by normalizing after each step (or using a parameterization)
    
    def _objective_and_grad(x: np.ndarray, beta: float) -> tuple[float, np.ndarray]:
        """Objective function for L-BFGS-B: minimize the surrogate."""
        pts = x.reshape(n, 3)
        
        # Normalize to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        
        value, grad = _log_sum_exp_surrogate(pts, beta)
        return value, grad
    
    # Multi-stage optimization with increasing beta
    current_points = best_points.copy()
    
    for beta in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline - 50:
            break
        
        # Define the objective for this beta
        def obj(x):
            val, grad = _objective_and_grad(x, beta)
            return val, grad
        
        # Run L-BFGS-B
        result = minimize(
            obj,
            current_points.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < _objective_and_grad(current_points.reshape(-1), beta)[0]:
            new_points = result.x.reshape(n, 3)
            # Normalize
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            new_points = new_points / norms
            
            score = _evaluate(new_points)
            if score > best_score:
                best_points = new_points.copy()
                best_score = score
                current_points = new_points.copy()
    
    # Phase 3: Fine-tuning with repulsion at small step sizes
    if time.time() < deadline - 20:
        points = _repulsion_optimize(best_points.copy(), n_iter=1000, lr=0.001)
        score = _evaluate(points)
        if score > best_score:
            best_points = points.copy()
            best_score = score
    
    # Final check
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(n)
        
    return best_points


# EVOLVE-BLOCK-END
