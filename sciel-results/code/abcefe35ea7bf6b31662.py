"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150

# EVOLVE-BLOCK-START
import numpy as np


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on the unit sphere.
    
    Energy = sum_{i<j} log(1/||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    Gradient with respect to x_i is projected onto the tangent space.
    """
    n = pts.shape[0]
    # Compute pairwise distances
    # pts: (N, 3)
    # dist: (N, N)
    dist = np.sqrt(np.maximum(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2), 1e-30))
    
    # Mask out diagonal (self-distances)
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum over i<j of log(dist_ij)
    # We sum over all i != j and divide by 2
    log_dists = np.log(dist)
    log_dists[~mask] = 0.0  # zero out diagonal
    energy = -0.5 * np.sum(log_dists)
    
    # Gradient:
    # d/dx_i [-log(||x_i - x_j||)] = (x_i - x_j) / ||x_i - x_j||^2
    # Total gradient for x_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    # But we need to be careful: energy = -sum_{i<j} log(dist_ij)
    # d/dx_i energy = -sum_{j != i} d/dx_i log(dist_ij) * (1/2 if we're summing over all pairs once)
    # Actually, let's think more carefully.
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j>i} d/dx_i log(d_ij) - sum_{j<i} d/dx_i log(d_ji)
    #         = -sum_{j != i} d/dx_i log(d_ij)  (since log(d_ij) = log(d_ji))
    # d/dx_i log(d_ij) = d/dx_i log(sqrt((x_i-x_j)^2)) = (x_i - x_j) / d_ij^2
    # So dE/dx_i = -sum_{j != i} (x_i - x_j) / d_ij^2 = sum_{j != i} (x_j - x_i) / d_ij^2
    
    # Compute gradient efficiently
    # For each pair (i,j), the contribution to grad_i is (x_j - x_i) / d_ij^2
    # and the contribution to grad_j is (x_i - x_j) / d_ij^2
    
    # Build the gradient matrix
    # inv_dist_sq: (N, N) with 1/d_ij^2 for i != j, 0 on diagonal
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # grad_i = sum_j (x_j - x_i) * inv_dist_sq[i,j]
    # = sum_j x_j * inv_dist_sq[i,j] - x_i * sum_j inv_dist_sq[i,j]
    # = (X @ inv_dist_sq)[i] - x_i * row_sum[i]
    
    X = pts  # (N, 3)
    inv_dist_sq_sym = inv_dist_sq  # already symmetric
    
    # sum_j x_j * inv_dist_sq[i,j] = inv_dist_sq @ X  (N x N @ N x 3 = N x 3)
    term1 = inv_dist_sq @ X  # (N, 3)
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    term2 = X * row_sums[:, np.newaxis]  # (N, 3)
    
    grad = term1 - term2  # (N, 3)
    
    # Project gradient onto tangent space: grad_tan = grad - (grad . x_i) * x_i
    dot_products = np.sum(grad * pts, axis=1)  # (N,)
    grad_tangent = grad - dot_products[:, np.newaxis] * pts  # (N, 3)
    
    return energy, grad_tangent


def _energy_only(pts):
    """Compute just the energy (for comparison with fsum)."""
    n = pts.shape[0]
    dist = np.sqrt(np.maximum(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :]) ** 2, axis=2), 1e-30))
    mask = ~np.eye(n, dtype=bool)
    # Use math.fsum for the final sum as specified
    log_dists = np.log(dist[mask])
    # We need sum of log(1/d) = -sum of log(d) for i<j
    # log_dists contains all i!=j, so we take half
    return -0.5 * math.fsum(log_dists.tolist())


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial: golden angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy_only(pts)
    
    # Try multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    def _run_optimization(init_pts, max_iters=500, tol=1e-10):
        """Run gradient-based optimization on the sphere."""
        pts = _project_to_sphere(init_pts.copy())
        current_energy = _energy_only(pts)
        
        # Use L-BFGS-B with a parameterization that keeps points on the sphere
        # We'll use a custom approach: optimize in R^3 and project back
        # Or better: use the fact that we can do gradient descent with re-projection
        
        # Actually, let's use scipy's minimize with a constraint-free approach
        # by optimizing over 2N parameters (spherical coordinates) or by
        # using the tangent space formulation.
        
        # Simpler approach: use L-BFGS-B on the 3N coordinates with the
        # understanding that we'll project back to the sphere at each step.
        # This isn't perfectly correct but works in practice for smooth objectives.
        
        # Better: use the fact that the gradient is already tangent, and do
        # gradient descent with adaptive step size, projecting back.
        
        lr = 0.01
        momentum = 0.0
        vel = np.zeros_like(pts)
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            energy, grad = _energy_and_grad(pts)
            
            # Adaptive learning rate
            # Scale gradient by typical norm
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            # Update with momentum
            vel = momentum * vel - lr * grad
            new_pts = pts + vel
            new_pts = _project_to_sphere(new_pts)
            
            # Check for improvement
            new_energy = _energy_only(new_pts)
            if new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                # Decrease momentum slightly as we converge
                momentum = 0.9 if it > 100 else 0.8
                # Reduce learning rate slowly
                if it % 50 == 0 and it > 0:
                    lr *= 0.9
            else:
                # Rejected step, reduce learning rate
                lr *= 0.5
                momentum = 0.0
                vel = np.zeros_like(pts)
            
            if it % 10 == 0 and it > 0:
                # Periodic check
                pass
        
        return pts, current_energy
    
    # Run optimization from the initial spiral
    pts_opt, energy_opt = _run_optimization(pts, max_iters=1000, tol=1e-10)
    if energy_opt < best_energy:
        best_pts = pts_opt
        best_energy = energy_opt
    
    # Try random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Random initial points
        random_pts = rng.randn(N, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        pts_opt, energy_opt = _run_optimization(random_pts, max_iters=500, tol=1e-10)
        if energy_opt < best_energy:
            best_pts = pts_opt
            best_energy = energy_opt
    
    # Final polish: run a few more iterations with smaller learning rate
    if time.time() < deadline - 5.0:
        pts_final, energy_final = _run_optimization(best_pts, max_iters=200, tol=1e-12)
        if energy_final < best_energy:
            best_pts = pts_final
            best_energy = energy_final
    
    # Ensure all points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    min_dist = np.min(np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2) * ~np.eye(N, dtype=bool))
    if min_dist < 1e-9:
        # Fall back to the spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return best_pts
# EVOLVE-BLOCK-END
