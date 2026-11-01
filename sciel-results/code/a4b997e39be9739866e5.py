"""Optimized logarithmic energy minimizer for 150 points on the sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _log_energy_and_grad(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|) and its gradient.
    
    Returns (energy, grad) where grad has shape (N, 3).
    """
    n = pts.shape[0]
    # Compute all pairwise distances
    # dist2[i,j] = |xi - xj|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(dist2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(n, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # We compute sum over all i!=j then divide by 2
    with np.errstate(divide='ignore'):
        log_d = np.where(mask, np.log(dist), 0.0)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient: d/dxi sum_{j!=i} log(1/|xi-xj|) = sum_{j!=i} (xi-xj)/|xi-xj|^2
    # Because d/dxi log(1/|xi-xj|) = d/dxi (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2 * (-1)
    # Wait: d/dxi (-log|xi-xj|) = -(1/|xi-xj|) * d|xi-xj|/dxi
    # d|xi-xj|/dxi = (xi-xj)/|xi-xj|
    # So d/dxi (-log|xi-xj|) = -(xi-xj)/|xi-xj|^2
    # And the energy is sum_{i<j} log(1/|xi-xj|), so gradient w.r.t. xi is:
    # sum_{j!=i} (xi-xj)/|xi-xj|^2  (each pair contributes to both points)
    # Actually for pair (i,j): contribution to grad_i = (xi-xj)/|xi-xj|^2, 
    # contribution to grad_j = (xj-xi)/|xi-xj|^2 = -(xi-xj)/|xi-xj|^2
    
    # grad[i] = sum_{j!=i} (xi - xj) / |xi - xj|^2
    # = sum_{j!=i} diff[i,j] / dist2[i,j]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # diff[i,j] = xi - xj
        # We want sum_j (xi-xj)/dist2[i,j] for j!=i
        inv_dist2 = np.where(mask, 1.0 / dist2, 0.0)
    
    grad = np.einsum('ij,ijk->ik', inv_dist2, diff)
    
    return energy, grad


def _log_energy_only(pts):
    """Just the energy value for evaluation."""
    energy, _ = _log_energy_and_grad(pts)
    return energy


def _project_to_sphere(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Initial points: Fibonacci spiral
    def make_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = make_fibonacci()
    best_energy = _log_energy_only(best_pts)
    
    # Try multiple random initializations and pick the best after a quick optimization
    np.random.seed(42)
    
    num_initials = 8
    for init_idx in range(num_initials):
        if time_left() < 30.0:
            break
        
        if init_idx == 0:
            pts = make_fibonacci()
        else:
            # Random points on sphere
            vecs = np.random.randn(N, 3)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            pts = vecs / norms
        
        # Quick L-BFGS optimization
        try:
            result = minimize(
                _log_energy_and_grad,
                pts,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8, 'maxls': 50}
            )
            opt_pts = _project_to_sphere(result.x)
            opt_energy = _log_energy_only(opt_pts)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Now do a more careful optimization from the best point found
    if time_left() > 20.0:
        try:
            result = minimize(
                _log_energy_and_grad,
                best_pts,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'maxls': 50}
            )
            opt_pts = _project_to_sphere(result.x)
            opt_energy = _log_energy_only(opt_pts)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Riemannian-style refinement: use the fact that we're on the sphere
    # Do a few rounds of projected gradient descent with line search
    if time_left() > 10.0:
        pts = best_pts.copy()
        lr = 0.01
        for iteration in range(200):
            if time_left() < 5.0:
                break
            
            energy, grad = _log_energy_and_grad(pts)
            
            # Project gradient to tangent space: grad_tangent = grad - (grad·x)x
            grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            
            # Gradient descent step
            new_pts = pts - lr * grad_tangent
            
            # Project back to sphere
            new_pts = _project_to_sphere(new_pts)
            
            new_energy = _log_energy_only(new_pts)
            
            if new_energy < energy:
                pts = new_pts
                # Adaptive learning rate
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pts = pts.copy()
            else:
                lr *= 0.5
            
            # Check convergence
            if np.linalg.norm(grad_tangent) < 1e-12:
                break
        
        # Final evaluation
        final_energy = _log_energy_only(best_pts)
        if final_energy < best_energy:
            best_energy = final_energy
    
    # Ensure all points are well-separated and on the sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)
    min_dist2 = np.min(dist2[~np.eye(N, dtype=bool)])
    
    if min_dist2 < 1e-18:
        # Fallback to Fibonacci
        best_pts = make_fibonacci()
    
    return best_pts
