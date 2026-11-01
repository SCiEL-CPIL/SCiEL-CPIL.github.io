"""Minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize, fmin_l_bfgs_b

N = 100
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 10.0  # seconds


def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient for points on the sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|)
    Gradient: -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    dist_sq = np.maximum(dist_sq, 1e-30)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Mask diagonal (self-distance)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # We sum over all i!=j and divide by 2
    log_dist = np.where(mask, np.log(dist), 0.0)
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient: for point i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = np.where(mask, 1.0 / dist_sq, 0.0)
    # grad_i = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = sum_j x_j * inv_dist_sq[i,j] - x_i * sum_j inv_dist_sq[i,j]
    sum_inv = np.sum(inv_dist_sq, axis=1)  # (n,)
    sum_x_weighted = inv_dist_sq @ pts  # (n, 3)
    grad = sum_x_weighted - pts * sum_inv[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _riemannian_descent(pts, max_iter=2000, tol=1e-12):
    """
    Riemannian gradient descent with backtracking line search.
    """
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    # Precompute initial energy
    energy, grad = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    for it in range(max_iter):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        # Tangential gradient: project gradient onto tangent space
        # tangent_grad = grad - (grad . x) * x
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        tangent_grad = grad - grad_dot_x * pts
        
        # Check convergence
        grad_norm = np.linalg.norm(tangent_grad)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        step_size = 1.0
        for _ in range(30):
            # Move in tangent direction
            new_pts = pts + step_size * tangent_grad
            new_pts = _project_to_sphere(new_pts)
            
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy < energy - 1e-4 * step_size * grad_norm * grad_norm:
                break
            step_size *= 0.5
        
        # Update
        pts = new_pts
        energy = new_energy
        grad = _energy_and_grad(pts)[1]
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    return best_pts, best_energy


def _lbfgs_optimize(pts, max_iter=1000):
    """
    L-BFGS-B optimization using scipy, with sphere constraint handled by
    projecting after each step.
    """
    def objective_and_grad(flat_pts):
        pts = flat_pts.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        energy, grad = _energy_and_grad(pts)
        # Project gradient to tangent space for proper optimization
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        tangent_grad = grad - grad_dot_x * pts
        return energy, tangent_grad.flatten()
    
    x0 = pts.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-14}
    )
    
    opt_pts = _project_to_sphere(result.x.reshape(-1, 3))
    opt_energy, _ = _energy_and_grad(opt_pts)
    return opt_pts, opt_energy


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def construct():
    start_time = time.time()
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts, best_energy = _energy_and_grad(pts)
    
    # Phase 1: Riemannian descent
    pts, energy = _riemannian_descent(pts, max_iter=1000, tol=1e-12)
    if energy < best_energy:
        best_energy = energy
        best_pts = pts.copy()
    
    # Phase 2: L-BFGS-B polish
    if time.time() < DEADLINE - SAFETY_MARGIN - 5:
        pts, energy = _lbfgs_optimize(pts, max_iter=500)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Phase 3: Multi-restart with different initializations
    restart_seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    for seed in restart_seeds:
        if time.time() > DEADLINE - SAFETY_MARGIN - 15:
            break
        
        # Try random initialization
        pts = _random_sphere(N, seed=seed)
        pts, energy = _riemannian_descent(pts, max_iter=500, tol=1e-10)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Polish with L-BFGS-B
        if time.time() < DEADLINE - SAFETY_MARGIN - 5:
            pts, energy = _lbfgs_optimize(pts, max_iter=300)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
    
    # Phase 4: Final high-precision polish on best solution
    if time.time() < DEADLINE - SAFETY_MARGIN:
        pts, energy = _lbfgs_optimize(best_pts, max_iter=1000)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Ensure all points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-8 * np.random.randn(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    # Final check
    final_pts = _project_to_sphere(best_pts)
    return final_pts
