"""Minimal logarithmic energy of 150 points on the sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # safety margin before 150s kill


def _energy_and_grad(pts):
    """
    Compute logarithmic energy E = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi-xj|)
    and its gradient w.r.t. pts (shape (N,3)).
    """
    # pts: (N,3)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    dist = np.linalg.norm(diff, axis=2)  # (N,N)
    # Avoid log(0)
    dist = np.maximum(dist, 1e-15)
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Because d/dx_i [-log|x_i - x_j|] = -(x_i - x_j)/|x_i - x_j|^2
    # So grad = -sum_{j != i} (x_i - x_j) / dist_ij^2
    # But energy = -sum log(dist), so dE/dx_i = -sum_{j!=i} (x_i - x_j)/dist^2
    # Wait: E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = -sum_{j != i} (1/2) * d(log d_ij)/dx_i ... let me be careful.
    # For pair (i,j): term is -log(d_ij) where d_ij = |x_i - x_j|
    # d(-log d_ij)/dx_i = -(1/d_ij) * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2
    # So grad_i = -sum_{j != i} (x_i - x_j)/d_ij^2
    # But this is for the unconstrained problem. Since we're on the sphere, we project.
    
    # Compute gradient efficiently
    # diff[i,j] = x_i - x_j
    # grad[i] = -sum_{j != i} diff[i,j] / dist[i,j]^2
    inv_dist2 = np.zeros_like(dist)
    mask = dist > 1e-15
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist2[i,j]
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize each point to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _spherical_energy_and_grad(theta_phi):
    """
    Energy and gradient in spherical coordinates.
    theta_phi: (N, 2) where col 0 = theta (polar), col 1 = phi (azimuthal)
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    pts = np.zeros((N, 3))
    pts[:, 0] = np.sin(theta) * np.cos(phi)
    pts[:, 1] = np.sin(theta) * np.sin(phi)
    pts[:, 2] = np.cos(theta)
    
    energy, grad_cart = _energy_and_grad(pts)
    
    # Convert Cartesian gradient to spherical gradient
    # dE/dtheta = grad_x * dx/dtheta + grad_y * dy/dtheta + grad_z * dz/dtheta
    # dx/dtheta = cos(theta)*cos(phi), dy/dtheta = cos(theta)*sin(phi), dz/dtheta = -sin(theta)
    # dE/dphi = grad_x * dx/dphi + grad_y * dy/dphi + grad_z * dz/dphi
    # dx/dphi = -sin(theta)*sin(phi), dy/dphi = sin(theta)*cos(phi), dz/dphi = 0
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    dtheta = np.zeros(N)
    dphi = np.zeros(N)
    
    dtheta = grad_cart[:, 0] * cos_t * cos_p + grad_cart[:, 1] * cos_t * sin_p + grad_cart[:, 2] * (-sin_t)
    dphi = grad_cart[:, 0] * (-sin_t * sin_p) + grad_cart[:, 1] * (sin_t * cos_p)
    
    grad_sph = np.column_stack([dtheta, dphi])
    
    return energy, grad_sph


def construct():
    t0 = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates
    theta0 = np.arccos(np.clip(pts0[:, 2], -1.0, 1.0))
    phi0 = np.arctan2(pts0[:, 1], pts0[:, 0])
    theta_phi0 = np.column_stack([theta0, phi0])
    
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(pts0)[0]
    
    # Try L-BFGS-B in spherical coordinates with multiple restarts
    rng = np.random.default_rng(42)
    
    num_restarts = 20
    max_time_per_opt = (DEADLINE - (time.time() - t0)) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 5:
            break
        
        if restart == 0:
            x0 = theta_phi0
        else:
            # Perturb the initial guess
            x0 = theta0.copy()
            x0[:, 0] += rng.normal(0, 0.5, N)
            x0[:, 1] += rng.normal(0, 0.5, N)
            x0[:, 0] = np.clip(x0[:, 0], 1e-6, np.pi - 1e-6)
            x0[:, 1] = x0[:, 1] % (2 * np.pi)
            x0 = np.column_stack([x0[:, 0], x0[:, 1]])
        
        # Set time limit for this optimization
        elapsed = time.time() - t0
        remaining = DEADLINE - 5 - elapsed
        if remaining <= 0:
            break
        
        # Use a callback to stop if time is running out
        state = {'stop': False}
        
        def callback(xk):
            if time.time() - t0 > DEADLINE - 5:
                state['stop'] = True
                return True
            return False
        
        try:
            result = minimize(
                _spherical_energy_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 10000,
                    'maxfun': 50000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'iprint': 0
                },
                callback=callback
            )
            
            if result.x is not None:
                theta = result.x[:, 0]
                phi = result.x[:, 1]
                pts = np.zeros((N, 3))
                pts[:, 0] = np.sin(theta) * np.cos(phi)
                pts[:, 1] = np.sin(theta) * np.sin(phi)
                pts[:, 2] = np.cos(theta)
                pts = _project_to_sphere(pts)
                
                # Check validity
                if np.all(np.isfinite(pts)) and np.all(np.linalg.norm(pts, axis=1) > 1e-10):
                    e, _ = _energy_and_grad(pts)
                    if np.isfinite(e) and e < best_energy:
                        best_energy = e
                        best_pts = pts
        except Exception:
            pass
    
    # Check for near-duplicate points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-9:
        # Fallback to original
        best_pts = pts0
    
    return best_pts
