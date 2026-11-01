"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _spherical_to_cartesian(params):
    """Convert spherical coordinates (theta, phi) to Cartesian coordinates on the unit sphere.
    params: array of shape (N, 2)
    Returns: array of shape (N, 3)
    """
    theta = params[:, 0]
    phi = params[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _log_energy_spherical(params):
    """Compute logarithmic energy from spherical coordinates."""
    points = _spherical_to_cartesian(params)
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    log_vals = np.log(d)
    return -math.fsum(list(log_vals))

def _log_energy_spherical_grad(params):
    """Compute gradient of logarithmic energy with respect to spherical coordinates."""
    theta = params[:, 0]
    phi = params[:, 1]
    points = _spherical_to_cartesian(params)
    n = len(points)
    
    # Precompute derivatives of points w.r.t theta and phi
    # d(x,y,z)/d(theta) = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # d(x,y,z)/d(phi)   = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    d_points_d_theta = np.column_stack([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ])
    
    d_points_d_phi = np.column_stack([
        -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.zeros(n)
    ])
    
    # Compute pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    
    dist = np.sqrt(dist_sq)
    
    # Gradient of energy E = - sum log(dist)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # But we need dE/dtheta_i and dE/dphi_i
    # dE/dtheta_i = sum_{j!=i} dE/dx_i . dx_i/dtheta_i
    
    # Vectorized calculation of sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Note: diff[i,j] = x_i - x_j
    # We want sum over j of (x_i - x_j) / dist_sq[i,j]
    
    # Mask diagonal to avoid self-interaction (though dist_sq diag is 1, diff diag is 0, so term is 0)
    # But to be safe, we can just compute it.
    
    # grad_cart[i] = sum_j (x_i - x_j) / |x_i - x_j|^2
    # Note: The term is (x_i - x_j) / dist_sq[i,j]
    
    # We need to be careful with the sign.
    # E = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
    # dE/dx_i = - sum_{j!=i} (1/d_ij) * (1/d_ij) * (x_i - x_j) / d_ij ? No.
    # d(log d)/dx_i = (1/d) * (x_i - x_j)/d = (x_i - x_j)/d^2
    # d(-log d)/dx_i = -(x_i - x_j)/d^2
    
    # So grad_cart[i] = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    
    # Let's compute sum_terms = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] is x_i - x_j.
    # dist_sq[i,j] is |x_i - x_j|^2.
    
    # We want to divide diff by dist_sq.
    # diff shape (N, N, 3), dist_sq shape (N, N)
    
    # To avoid division by zero, we already set diagonal dist_sq to 1. diff diagonal is 0.
    # So term is 0.
    
    # However, we need to sum over j for each i.
    # grad_cart = - np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Wait, is it sum over all j or just j>i?
    # The gradient of the total energy with respect to x_i involves all j != i.
    # The term for pair (i,j) contributes to grad_i and grad_j.
    # In the sum over pairs (i<j), the term is log(1/d_ij).
    # dE/dx_i = sum_{j>i} d/dx_i log(1/d_ij) + sum_{j<i} d/dx_i log(1/d_ji)
    # Since d_ij = d_ji, this is sum_{j!=i} d/dx_i log(1/d_ij)
    # = sum_{j!=i} -(x_i - x_j)/|x_i - x_j|^2
    
    # So the formula holds.
    
    grad_cart = -np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Now project to spherical coordinates
    # dE/dtheta_i = grad_cart[i] . d_points_d_theta[i]
    # dE/dphi_i = grad_cart[i] . d_points_d_phi[i]
    
    grad_theta = np.sum(grad_cart * d_points_d_theta, axis=1)
    grad_phi = np.sum(grad_cart * d_points_d_phi, axis=1)
    
    return np.column_stack([grad_theta, grad_phi])

def construct():
    # Initial configuration: Fibonacci spiral in Cartesian, convert to Spherical
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_init = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates
    theta_init = np.arccos(np.clip(points_init[:, 2], -1.0, 1.0))
    phi_init = np.arctan2(points_init[:, 1], points_init[:, 0])
    
    params_init = np.column_stack([theta_init, phi_init])
    
    # Optimize using L-BFGS-B
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi), (0, 2 * np.pi)] * N
    
    result = minimize(
        _log_energy_spherical,
        params_init,
        method='L-BFGS-B',
        jac=_log_energy_spherical_grad,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50}
    )
    
    # Get the optimized parameters
    params_opt = result.x
    
    # Convert back to Cartesian
    points = _spherical_to_cartesian(params_opt)
    
    # Normalize to be safe
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Check for close points
    min_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_dist:
                min_dist = d
                
    if min_dist < 1e-9:
        # Perturb if necessary
        rng = np.random.default_rng(42)
        points = points + rng.normal(0, 1e-6, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points
# EVOLVE-BLOCK-END
