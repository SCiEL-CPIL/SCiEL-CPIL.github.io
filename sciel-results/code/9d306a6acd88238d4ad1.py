"""
Stronger algorithm for 100-point logarithmic energy on the unit sphere.
Uses a multi-stage approach:
1. Riemannian Gradient Descent for coarse optimization.
2. L-BFGS-B for fine optimization.
3. Multiple restarts with different initializations.
4. Final ultra-high-precision polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
# Hard wall-clock deadline with safety margin (150s limit, use 145s)
DEADLINE = time.time() + 145.0


def _time_left():
    return max(0.0, DEADLINE - time.time())


def _project(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1e-15, norms)
    return points / norms


def _energy_and_grad(points):
    """
    Calculate logarithmic energy and its gradient.
    E = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    grad_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = points.shape[0]
    # Compute pairwise differences
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    # Compute squared distances
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid log(0)
    
    # Energy calculation
    # We sum over i < j. The matrix is symmetric, so we can take the upper triangle.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_sq = dist_sq[mask]
    # Ensure no zeros (shouldn't happen with projection, but safe)
    d_sq = np.maximum(d_sq, 1e-30)
    energy = -0.5 * np.sum(np.log(d_sq))
    
    # Gradient calculation
    # grad_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: The factor 0.5 in energy comes from double counting pairs.
    # Let's verify the gradient factor.
    # E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_i = - sum_{j!=i, j>i} (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j|| - sum_{j!=i, j<i} (1/||x_j - x_i||) * (-(x_i - x_j))/||x_j - x_i||
    # Since ||x_i - x_j|| = ||x_j - x_i||, the terms are:
    # - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # So the gradient is exactly -sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # In the vectorized form:
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.einsum('ij,ijk->ik', inv_dist_sq, diff)
    
    # Project gradient to tangent space (optional for ambient optimization, but good for Riemannian)
    # For L-BFGS-B on the sphere, we usually just use the ambient gradient and project the result.
    # However, using the projected gradient can be more stable.
    g_tangent = grad - (grad * points).sum(axis=1, keepdims=True) * points
    
    return energy, g_tangent


def _riemannian_flow(points, time_budget, initial_lr=0.05, min_lr=1e-8, patience=500):
    """
    Riemannian Gradient Descent with backtracking line search.
    """
    start_time = time.time()
    best_points = points.copy()
    best_energy, _ = _energy_and_grad(best_points)
    lr = initial_lr
    no_improve = 0
    
    while time.time() - start_time < time_budget and _time_left() > 1.0:
        energy, grad = _energy_and_grad(best_points)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            # Riemannian step: move in direction -grad, then re-project
            # We use a simple projection step. A retraction like expmap is better but slower.
            new_points = best_points - alpha * grad
            new_points = _project(new_points)
            
            new_energy, _ = _energy_and_grad(new_points)
            
            if new_energy < energy - 1e-12:
                best_points = new_points
                best_energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if improved:
            no_improve = 0
            # Optionally decay lr slowly
            # lr *= 0.999
        else:
            no_improve += 1
            lr *= 0.9
            if no_improve > patience:
                break
                
    return best_points


def _lbfgs_polish(points, time_budget):
    """
    Polish using L-BFGS-B.
    """
    start_time = time.time()
    
    def fun_and_grad(x):
        p = _project(x.reshape(-1, 3))
        e, g = _energy_and_grad(p)
        # Return ambient gradient for L-BFGS-B
        # The gradient of the projected function is not exactly g, but for small steps it's a good approximation.
        # To be more rigorous, one should use the projected gradient.
        return e, g.flatten()
    
    x0 = points.flatten().copy()
    
    try:
        # Use tight tolerances
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100, 'maxfun': 100000})
        if hasattr(res, 'x') and res.success:
            new_points = _project(res.x.reshape(-1, 3))
            return new_points
    except Exception:
        pass
    
    return points


def _fibonacci_spiral(n):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project(points)


def _random_points(n, seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    return _project(points)


def _deduplicate(points, min_dist=1e-9):
    """Ensure no points are closer than min_dist."""
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                # Perturb point i
                noise = np.random.randn(3) * 1e-6
                points[i] = _project(points[i] + noise)
    return points


def construct():
    global DEADLINE
    DEADLINE = time.time() + 145.0
    
    best_points = None
    best_energy = float('inf')
    
    # Phase 1: Fibonacci Spiral Start
    points_fib = _fibonacci_spiral(N)
    points_fib = _deduplicate(points_fib)
    
    # Coarse Riemannian Flow
    budget1 = min(_time_left() * 0.2, 10.0)
    if budget1 > 0.5:
        refined = _riemannian_flow(points_fib, budget1)
        e, _ = _energy_and_grad(refined)
        if e < best_energy:
            best_energy = e
            best_points = refined
    
    # Fine L-BFGS Polish
    if best_points is not None and _time_left() > 5.0:
        budget2 = min(_time_left() * 0.3, 20.0)
        polished = _lbfgs_polish(best_points, budget2)
        e, _ = _energy_and_grad(polished)
        if e < best_energy:
            best_energy = e
            best_points = polished
    
    # Phase 2: Random Restarts
    # We have remaining time for random restarts
    restart_time_limit = _time_left() * 0.6
    restart_start = time.time()
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    
    for seed in seeds:
        if time.time() - restart_start > restart_time_limit:
            break
        if _time_left() < 10.0:
            break
        
        random_points = _random_points(N, seed)
        random_points = _deduplicate(random_points)
        
        # Coarse Riemannian Flow
        budget_coarse = min(2.0, _time_left() * 0.1)
        if budget_coarse < 0.2:
            break
            
        refined = _riemannian_flow(random_points, budget_coarse, initial_lr=0.1, patience=200)
        e, _ = _energy_and_grad(refined)
        
        if e < best_energy:
            best_energy = e
            best_points = refined
            
            # If we found a significantly better solution, polish it immediately
            if _time_left() > 5.0:
                budget_polish = min(5.0, _time_left() * 0.2)
                polished = _lbfgs_polish(best_points, budget_polish)
                e_polished, _ = _energy_and_grad(polished)
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_points = polished
    
    # Phase 3: Final Ultra-High-Precision Polish
    if best_points is not None and _time_left() > 2.0:
        # Use the remaining time for a final polish
        final_polish = _lbfgs_polish(best_points, _time_left() - 1.0)
        e_final, _ = _energy_and_grad(final_polish)
        if e_final < best_energy:
            best_energy = e_final
            best_points = final_polish
    
    # Final Validation and Deduplication
    if best_points is None:
        # Fallback
        best_points = _fibonacci_spiral(N)
        
    best_points = _deduplicate(best_points)
    best_points = _project(best_points)
    
    return best_points
