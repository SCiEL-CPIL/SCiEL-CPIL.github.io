"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient on the sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient on the sphere (tangent component).
    """
    # pts: (N, 3)
    # Compute all pairwise distances
    # d_ij = |x_i - x_j|
    # We use the formula: d_ij^2 = 2 - 2*x_i·x_j for unit vectors
    dots = pts @ pts.T  # (N, N)
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)  # avoid division by zero on diagonal
    d = np.sqrt(d2)
    # Avoid log(0) issues
    d = np.maximum(d, 1e-15)
    
    # Energy: sum over i<j of -log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2 * (1) ... 
    # Actually, E = -sum_{i<j} log|d_ij|
    # dE/dx_i = sum_{j != i} (1/|d_ij|) * (x_i - x_j)/|d_ij| = sum_{j != i} (x_i - x_j)/|d_ij|^2
    # But we need the tangent component on the sphere.
    # The gradient in R^3 is g_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # The Riemannian gradient (tangent) is g_i - (g_i · x_i) x_i
    
    # Compute gradient in R^3
    # grad[i] = sum_j (x_i - x_j) / d_ij^2 for j != i
    # (x_i - x_j) / d_ij^2 = (x_i - x_j) / (2 - 2*x_i·x_j)
    # Let's compute this efficiently
    # inv_d2 = 1/d2, with diagonal set to 0
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) * inv_d2[i,j]
    # = x_i * sum_j inv_d2[i,j] - sum_j x_j * inv_d2[i,j]
    row_sums = inv_d2.sum(axis=1)  # (N,)
    # sum_j x_j * inv_d2[i,j] = inv_d2 @ pts
    weighted_sum = inv_d2 @ pts  # (N, 3)
    grad = pts * row_sums[:, np.newaxis] - weighted_sum  # (N, 3)
    
    # Project to tangent space
    grad_dot_x = np.sum(grad * pts, axis=1)  # (N,)
    grad_tangent = grad - grad_dot_x[:, np.newaxis] * pts
    
    return energy, grad_tangent


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _check_validity(pts):
    """Check if points are valid: shape, non-zero, finite, no too-close pairs."""
    if pts.shape != (N, 3):
        return False
    if not np.all(np.isfinite(pts)):
        return False
    if np.any(np.linalg.norm(pts, axis=1) < 1e-10):
        return False
    # Check minimum distance
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    min_d2 = d2.min()
    if min_d2 < (1e-9) ** 2:
        return False
    return True


def construct():
    deadline = time.time() + 140.0  # 140s to be safe (150s limit)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_energy = energy
    
    # Use L-BFGS-B with manual projection? No, better to use a method that handles constraints.
    # We'll use scipy.optimize.minimize with BFGS and project back to sphere after each step.
    # Or better: use the fact that we can optimize on the sphere using Riemannian gradient.
    # scipy doesn't natively support Riemannian optimization, so we'll do manual gradient descent
    # with projection, or use a parameterization.
    
    # Alternative: use L-BFGS-B with the constraint that |x_i| = 1 via penalty or by 
    # re-parameterizing. Let's try a simple approach: gradient descent with projection.
    
    # Actually, let's use scipy.optimize.minimize with BFGS and a custom approach:
    # We'll minimize in R^3 with the constraint that each point is on the unit sphere.
    # We can use SLSQP with constraints, but that's slow.
    
    # Better approach: Use the Riemannian gradient and do line search on the sphere.
    # Or: use scipy's minimize with method='L-BFGS-B' on a re-parameterization.
    
    # Let's try a practical approach: use scipy.optimize.minimize with BFGS,
    # and after each iteration, project back to the sphere. This is a common heuristic.
    
    # Even better: use the fact that for points on the sphere, we can use
    # the gradient in the ambient space and project. Let's implement a simple
    # Riemannian gradient descent with backtracking line search.
    
    def _riemannian_line_search(pts, grad_tangent, energy, max_iter=50):
        """Backtracking line search on the sphere."""
        # Initial step size
        step = 0.01
        for _ in range(max_iter):
            # Move in tangent direction
            new_pts = pts + step * grad_tangent
            # Project back to sphere
            new_pts = _normalize(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                return new_pts, new_energy, step
            step *= 0.5
        return pts, energy, step
    
    # Do Riemannian gradient descent with adaptive step size
    step_size = 0.05
    for it in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        energy, grad_tangent = _energy_and_grad(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-10:
            break
        
        # Riemannian gradient step with line search
        new_pts, new_energy, used_step = _riemannian_line_search(pts, grad_tangent, energy)
        
        if new_energy < energy:
            pts = new_pts
            # Adapt step size
            if used_step < step_size:
                step_size = max(used_step, 0.001)
            else:
                step_size = min(step_size * 1.1, 0.5)
        else:
            step_size *= 0.5
    
    # Try to improve with scipy's minimize using a different starting point
    if time.time() < deadline - 20.0:
        # Use L-BFGS-B with a penalty for non-unit norms
        def _objective_and_grad_flat(x):
            p = x.reshape(N, 3)
            p = _normalize(p)
            e, g = _energy_and_grad(p)
            # Gradient with respect to x (before normalization)
            # This is complex, so let's just use the gradient on the sphere
            # and hope L-BFGS-B can handle it with the normalization
            return e, g.reshape(-1)
        
        # Try from the current best
        try:
            res = minimize(
                _objective_and_grad_flat,
                best_pts.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            res_pts = _normalize(res.x.reshape(N, 3))
            res_energy, _ = _energy_and_grad(res_pts)
            if res_energy < best_energy and _check_validity(res_pts):
                best_pts = res_pts
                best_energy = res_energy
        except Exception:
            pass
    
    # Final polish with smaller steps
    if time.time() < deadline - 5.0:
        pts = best_pts.copy()
        step_size = 0.001
        for it in range(2000):
            if time.time() > deadline - 2.0:
                break
            energy, grad_tangent = _energy_and_grad(pts)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
            grad_norm = np.linalg.norm(grad_tangent)
            if grad_norm < 1e-12:
                break
            new_pts, new_energy, used_step = _riemannian_line_search(pts, grad_tangent, energy, max_iter=30)
            if new_energy < energy:
                pts = new_pts
            else:
                step_size *= 0.5
    
    # Ensure validity
    best_pts = _normalize(best_pts)
    if not _check_validity(best_pts):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
