"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy_and_grad(x):
    """
    Computes the logarithmic energy and its gradient for points on the sphere.
    x is (N, 3) array of points on the unit sphere.
    Energy: E = sum_{i<j} log(1/||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    Gradient: grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # x is (N, 3)
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist is (N, N)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    # sum_{i<j} log(1/dist) = 0.5 * sum_{i != j} log(1/dist)
    # log(1/d) = -log(d)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _stereo_params_to_sphere(params):
    """
    Convert stereographic projection parameters (u, v) to points on the sphere.
    Standard stereographic projection from (0,0,-1) to plane z=0.
    x = 2u / (1 + u^2 + v^2)
    y = 2v / (1 + u^2 + v^2)
    z = (1 - u^2 - v^2) / (1 + u^2 + v^2)
    """
    u = params[0::2]
    v = params[1::2]
    denom = 1.0 + u**2 + v**2
    x = 2.0 * u / denom
    y = 2.0 * v / denom
    z = (1.0 - u**2 - v**2) / denom
    return np.column_stack([x, y, z])

def _sphere_to_stereo_params(points):
    """
    Inverse stereographic projection.
    u = x / (1 - z)
    v = y / (1 - z)
    """
    eps = 1e-9
    z_safe = np.where(points[:, 2] < 1.0 - eps, points[:, 2], 1.0 - eps)
    u = points[:, 0] / (1.0 - z_safe)
    v = points[:, 1] / (1.0 - z_safe)
    params = np.zeros(2 * N)
    params[0::2] = u
    params[1::2] = v
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_x = None
    best_energy = float('inf')
    
    # Generate multiple initial guesses
    # 1. Fibonacci sphere
    # 2-5. Random points (with some spread)
    initial_guesses = [_fibonacci_sphere(N)]
    rng = np.random.default_rng(42)
    for _ in range(4):
        # Random points, normalized
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        initial_guesses.append(pts)
        
    for x0 in initial_guesses:
        if time.time() > deadline:
            break
            
        # Convert to stereographic parameters
        # Check if any point is too close to the south pole (0,0,-1)
        # Stereographic projection from south pole is singular at south pole.
        # If a point is close to (0,0,-1), u,v become large.
        # This is numerically okay if we use double precision, but let's check.
        # Actually, the standard map is from (0,0,-1) to z=0 plane.
        # The point (0,0,-1) maps to infinity.
        # If our points are well distributed, this is fine.
        
        p0 = _sphere_to_stereo_params(x0)
        
        # Define objective and gradient in parameter space
        def obj_and_grad(params):
            x = _stereo_params_to_sphere(params)
            # Ensure points are on sphere (they should be by construction)
            # Compute energy and gradient in R^3
            e, grad_r3 = _log_energy_and_grad(x)
            
            # Chain rule: grad_params = J^T grad_r3
            # x = (2u/d, 2v/d, (1-u^2-v^2)/d) where d = 1+u^2+v^2
            # Let's compute Jacobian numerically or analytically?
            # Analytic is better for speed and accuracy.
            
            u = params[0::2]
            v = params[1::2]
            d = 1.0 + u**2 + v**2
            
            # dx/du, dx/dv
            # x = 2u/d
            # dx/du = (2d - 2u*2u)/d^2 = 2(1+u^2+v^2 - 2u^2)/d^2 = 2(1 - u^2 + v^2)/d^2
            # dx/dv = -2u*2v/d^2 = -4uv/d^2
            # y = 2v/d
            # dy/du = -4uv/d^2
            # dy/dv = 2(1 + u^2 - v^2)/d^2
            # z = (1 - u^2 - v^2)/d
            # dz/du = (-2u*d - (1-u^2-v^2)*2u)/d^2 = -2u(d + 1 - u^2 - v^2)/d^2 = -2u(2)/d^2 = -4u/d^2
            # dz/dv = -4v/d^2
            
            d_inv2 = 1.0 / (d * d)
            
            dx_du = 2.0 * (1.0 - u**2 + v**2) * d_inv2
            dx_dv = -4.0 * u * v * d_inv2
            dy_du = -4.0 * u * v * d_inv2
            dy_dv = 2.0 * (1.0 + u**2 - v**2) * d_inv2
            dz_du = -4.0 * u * d_inv2
            dz_dv = -4.0 * v * d_inv2
            
            # grad_params[2i] = grad_x_i * dx_du + grad_y_i * dy_du + grad_z_i * dz_du
            # grad_params[2i+1] = grad_x_i * dx_dv + grad_y_i * dy_dv + grad_z_i * dz_dv
            
            grad_params = np.zeros(2 * N)
            grad_params[0::2] = grad_r3[:, 0] * dx_du + grad_r3[:, 1] * dy_du + grad_r3[:, 2] * dz_du
            grad_params[1::2] = grad_r3[:, 0] * dx_dv + grad_r3[:, 1] * dy_dv + grad_r3[:, 2] * dz_dv
            
            return e, grad_params
        
        # Run L-BFGS-B
        # maxiter is limited by time, but L-BFGS-B is fast.
        try:
            res = minimize(obj_and_grad, p0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-8})
            if res.success or res.status == 0:
                p_opt = res.x
                x_opt = _stereo_params_to_sphere(p_opt)
                e_opt, _ = _log_energy_and_grad(x_opt)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_x = x_opt.copy()
        except Exception:
            pass
        
    # If no optimization succeeded, fallback to Fibonacci
    if best_x is None:
        best_x = _fibonacci_sphere(N)
        
    # Final normalization to be safe
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x

# EVOLVE-BLOCK-END
