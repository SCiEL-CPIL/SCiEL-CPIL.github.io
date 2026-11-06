import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_grad_cartesian(points):
    """
    Compute the logarithmic energy and its Cartesian gradient for points on the sphere.
    Energy E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    Gradient for point i: grad_i = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # points shape (N, 3)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    mask = ~np.eye(N, dtype=bool)
    
    log_d = np.zeros_like(dist_sq)
    log_d[mask] = -np.log(dist[mask])
    energy = 0.5 * np.sum(log_d)
    
    inv_dist_sq = 1.0 / dist_sq
    terms = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(terms, axis=1)
    
    return energy, grad

def _points_to_spherical(points):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta is polar angle [0, pi]
    theta = np.arccos(z / r)
    # phi is azimuthal angle [0, 2pi)
    phi = np.arctan2(y, x)
    # Ensure phi is in [0, 2pi)
    phi = phi % (2 * np.pi)
    return theta, phi

def _spherical_to_points(theta, phi):
    """Convert spherical coordinates to Cartesian points."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])

def _energy_and_grad_spherical(theta_phi):
    """
    Objective and gradient for optimization in spherical coordinates.
    theta_phi: array of shape (2*N,)
    """
    N_opt = theta_phi.shape[0] // 2
    theta = theta_phi[:N_opt]
    phi = theta_phi[N_opt:]
    
    points = _spherical_to_points(theta, phi)
    
    # Calculate energy and Cartesian gradient
    energy, grad_cart = _energy_and_grad_cartesian(points)
    
    # Calculate gradient with respect to theta and phi
    # Chain rule: dE/dtheta = sum_k dE/dx_k * dx_k/dtheta
    # dx/dtheta = [cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)]
    # dx/dphi = [-sin(theta)sin(phi), sin(theta)cos(phi), 0]
    
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    dxdtheta = np.column_stack([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta])
    dxdphi = np.column_stack([-sin_theta * sin_phi, sin_theta * cos_phi, np.zeros(N_opt)])
    
    grad_theta = np.sum(grad_cart * dxdtheta, axis=1)
    grad_phi = np.sum(grad_cart * dxdphi, axis=1)
    
    grad_sph = np.concatenate([grad_theta, grad_phi])
    
    return energy, grad_sph

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points_fib = points_fib / np.linalg.norm(points_fib, axis=1, keepdims=True)
    
    best_points = points_fib.copy()
    best_energy = _energy_and_grad_cartesian(points_fib)[0]
    
    # 2. Optimization using L-BFGS-B in Spherical Coordinates
    # We try a few starts: Fibonacci and a couple of random ones if time permits.
    
    starts = [points_fib]
    
    # Add random starts if time allows
    if time.time() < start_time + 10:
        for _ in range(2):
            pts = np.random.randn(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            starts.append(pts)
    
    for start_pts in starts:
        if time.time() > deadline - 10:
            break
            
        theta, phi = _points_to_spherical(start_pts)
        x0 = np.concatenate([theta, phi])
        
        # Optimization options
        options = {
            'maxiter': 500,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 50
        }
        
        try:
            res = minimize(
                _energy_and_grad_spherical,
                x0,
                jac=True,
                method='L-BFGS-B',
                options=options
            )
            
            if res.success or res.fun < best_energy:
                theta_opt, phi_opt = res.x[:N], res.x[N:]
                # Clamp theta to avoid numerical issues at poles
                theta_opt = np.clip(theta_opt, 1e-6, np.pi - 1e-6)
                new_points = _spherical_to_points(theta_opt, phi_opt)
                # Normalize just in case
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_energy = _energy_and_grad_cartesian(new_points)[0]
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_points = new_points.copy()
                    
        except Exception as e:
            pass
            
    # 3. Polish Phase: High precision local optimization
    # Use the best points found so far
    if time.time() < deadline - 5:
        theta, phi = _points_to_spherical(best_points)
        x0 = np.concatenate([theta, phi])
        
        options = {
            'maxiter': 1000,
            'ftol': 1e-15,
            'gtol': 1e-10,
            'maxls': 100
        }
        
        try:
            res = minimize(
                _energy_and_grad_spherical,
                x0,
                jac=True,
                method='L-BFGS-B',
                options=options
            )
            
            if res.fun < best_energy:
                theta_opt, phi_opt = res.x[:N], res.x[N:]
                theta_opt = np.clip(theta_opt, 1e-6, np.pi - 1e-6)
                new_points = _spherical_to_points(theta_opt, phi_opt)
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_energy = _energy_and_grad_cartesian(new_points)[0]
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_points = new_points.copy()
        except Exception as e:
            pass

    # Final validation
    if not np.all(np.isfinite(best_points)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_points
