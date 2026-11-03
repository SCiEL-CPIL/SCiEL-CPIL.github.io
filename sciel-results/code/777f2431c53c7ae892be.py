"""Improved logarithmic energy minimizer using gradient descent with momentum and L-BFGS-B polish."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
DEADLINE = 140.0  # Safety margin before 150s kill

def _log_energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    # Compute all pairwise distances
    # points shape: (n, 3)
    # dist^2[i,j] = |x_i - x_j|^2
    # Use matrix form: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    # Since points are on unit sphere, ||x_i||^2 = 1
    gram = points @ points.T
    dist_sq = 1.0 + 1.0 - 2.0 * gram
    np.fill_diagonal(dist_sq, 1.0)
    # Clip to avoid log(0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    # We need sum over i<j of log(1/sqrt(dist_sq)) = -0.5 * sum_{i<j} log(dist_sq)
    # Take upper triangle
    iu = np.triu_indices(n, k=1)
    d_sq_pairs = dist_sq[iu]
    # log(1/d) where d = sqrt(dist_sq), so log(1/d) = -0.5 * log(dist_sq)
    return -0.5 * math.fsum(np.log(d_sq_pairs))

def _log_energy_grad(points):
    """Compute gradient of logarithmic energy w.r.t. points.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    d/dx_i of -log(|x_i - x_j|) = -(x_i - x_j)/|x_i - x_j|^2
    So grad_i = sum_{j != i} (x_i - x_j)/|x_i - x_j|^2 ... wait let me re-derive.
    
    E = sum_{i<j} log(1/d_ij) where d_ij = |x_i - x_j|
    E = -sum_{i<j} log(d_ij)
    dE/dx_i = -sum_{j != i} (1/d_ij) * d(d_ij)/dx_i
    d(d_ij)/dx_i = (x_i - x_j)/d_ij
    So dE/dx_i = -sum_{j != i} (x_i - x_j)/d_ij^2 = sum_{j != i} (x_j - x_i)/d_ij^2
    """
    n = len(points)
    gram = points @ points.T
    dist_sq = 1.0 + 1.0 - 2.0 * gram
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # grad_i = sum_{j != i} (x_j - x_i) / dist_sq_ij
    # = sum_j (x_j - x_i)/dist_sq_ij - (x_i - x_i)/dist_sq_ii (but we set diag to 1, so need to handle)
    # Actually: sum_{j != i} (x_j - x_i) / dist_sq_ij
    # = sum_j (x_j / dist_sq_ij) - x_i * sum_j (1/dist_sq_ij)
    # But for j=i, dist_sq_ii = 1 (we set it), and (x_i - x_i) = 0, so the term is 0.
    # So we can include j=i in the sum since the numerator is 0.
    
    inv_dist_sq = 1.0 / dist_sq
    # Set diagonal to 0 to exclude self
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_j inv_dist_sq[i,j] * (x_j - x_i)
    #        = sum_j inv_dist_sq[i,j] * x_j - x_i * sum_j inv_dist_sq[i,j]
    row_sum = inv_dist_sq.sum(axis=1)
    grad = inv_dist_sq @ points - row_sum[:, np.newaxis] * points
    
    return grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _golden_spiral():
    """Generate initial points using golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    
    # Initial points
    pts = _golden_spiral()
    pts = _project_to_sphere(pts)
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Phase 1: Gradient descent with momentum
    lr = 0.01
    momentum = 0.9
    velocity = np.zeros_like(pts)
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        grad = _log_energy_grad(pts)
        
        # Project gradient to tangent space
        # Tangent space at x_i: vectors v such that x_i . v = 0
        # proj(v) = v - (x_i . v) * x_i
        tangent_grad = grad - (pts * grad).sum(axis=1, keepdims=True) * pts
        
        velocity = momentum * velocity - lr * tangent_grad
        pts = pts + velocity
        pts = _project_to_sphere(pts)
        
        energy = _log_energy(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.8
    
    # Phase 2: L-BFGS-B polish using spherical coordinates
    if time.time() - start_time < DEADLINE - 20:
        # Convert to (theta, phi) coordinates for optimization
        # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
        def pts_to_params(p):
            theta = np.arccos(np.clip(p[:, 2], -1, 1))
            phi = np.arctan2(p[:, 1], p[:, 0])
            return np.concatenate([theta, phi])
        
        def params_to_pts(params):
            theta = params[:N]
            phi = params[N:]
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            return np.column_stack([x, y, z])
        
        def obj_and_grad(params):
            p = params_to_pts(params)
            e = _log_energy(p)
            # Compute gradient in parameter space via chain rule
            # This is complex; let's use numerical gradient or just L-BFGS-B with numerical grad
            return e
        
        initial_params = pts_to_params(best_pts)
        
        try:
            result = minimize(
                obj_and_grad,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-10}
            )
            if result.success or result.fun < best_energy:
                polished_pts = params_to_pts(result.x)
                polished_pts = _project_to_sphere(polished_pts)
                polished_energy = _log_energy(polished_pts)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts
        except Exception:
            pass
    
    # Final check
    if time.time() - start_time < DEADLINE - 5:
        # One more quick polish with smaller steps
        pts2 = best_pts.copy()
        lr2 = 0.001
        vel2 = np.zeros_like(pts2)
        for _ in range(200):
            if time.time() - start_time > DEADLINE - 5:
                break
            grad2 = _log_energy_grad(pts2)
            tangent_grad2 = grad2 - (pts2 * grad2).sum(axis=1, keepdims=True) * pts2
            vel2 = 0.9 * vel2 - lr2 * tangent_grad2
            pts2 = pts2 + vel2
            pts2 = _project_to_sphere(pts2)
            e2 = _log_energy(pts2)
            if e2 < best_energy:
                best_energy = e2
                best_pts = pts2.copy()
    
    return _project_to_sphere(best_pts)
