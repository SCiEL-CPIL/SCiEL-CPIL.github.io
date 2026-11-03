import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _energy_and_grad(params):
    # params: (N, 2) theta, phi
    theta = params[:, 0]
    phi = params[:, 1]
    
    # Convert to Cartesian
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack((x, y, z))
    
    # Pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dist_sq, 1.0) # avoid log(0) on diagonal, we'll ignore them
    
    # Regularized distance squared
    eps = 1e-6
    dist_sq_reg = dist_sq + eps**2
    
    # Energy: -0.5 * sum(log(dist_sq_reg)) for i < j
    # Sum over all i != j is 2 * sum over i < j
    # So sum over i<j is 0.5 * sum_{i != j}
    log_dist_sq = np.log(dist_sq_reg)
    # Set diagonal to 0
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.5 * np.sum(log_dist_sq)
    
    # Gradient
    # grad_x[i] = sum_{j != i} (pts[i] - pts[j]) / dist_sq_reg[i, j]
    # Note: The gradient of -0.5 log(d^2) is -0.5 * (2/d^2) * (x_i - x_j) = -(x_i - x_j)/d^2
    # Wait, E = - sum log(d) = -0.5 sum log(d^2).
    # dE/dx_i = -0.5 * sum_{j} d/dx_i log(d_ij^2)
    # d/dx_i log((x_i - x_j)^2) = 2 (x_i - x_j) / (x_i - x_j)^2 = 2 (x_i - x_j) / d^2
    # So dE/dx_i = -0.5 * sum_{j} 2 (x_i - x_j) / d^2 = - sum_{j} (x_i - x_j) / d^2
    # This is the force on i.
    
    # Using regularized dist_sq
    # term[i, j] = (pts[i] - pts[j]) / dist_sq_reg[i, j]
    # We need to be careful with broadcasting.
    
    # dist_sq_reg is (N, N)
    # diff is (N, N, 3)
    # We want to sum over j for each i.
    # coeff[i, j, k] = diff[i, j, k] / dist_sq_reg[i, j]
    # But we only want i != j.
    
    coeff = diff / dist_sq_reg[:, :, np.newaxis]
    # Zero out diagonal
    coeff[np.arange(N), np.arange(N), :] = 0.0
    
    # grad_x[i] = - sum_j coeff[i, j]
    grad_x = -np.sum(coeff, axis=1)
    
    # Now map grad_x to grad_theta, grad_phi
    # grad_theta[i] = dot(grad_x[i], d(pts[i])/d(theta[i]))
    # grad_phi[i] = dot(grad_x[i], d(pts[i])/d(phi[i]))
    
    d_pts_d_theta = np.column_stack((
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        -np.sin(theta)
    ))
    
    d_pts_d_phi = np.column_stack((
        -np.sin(theta) * np.sin(phi),
        np.sin(theta) * np.cos(phi),
        np.zeros(N)
    ))
    
    grad_theta = np.sum(grad_x * d_pts_d_theta, axis=1)
    grad_phi = np.sum(grad_x * d_pts_d_phi, axis=1)
    
    grad = np.column_stack((grad_theta, grad_phi))
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s buffer
    
    best_energy = float('inf')
    best_params = None
    
    # Initial guesses
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    theta_fib = np.arccos(z)
    phi_fib = np.pi * (1.0 + np.sqrt(5.0)) * i
    init_fib = np.column_stack((theta_fib, phi_fib))
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.uniform(0, np.pi, (N, 1))
    init_rand = np.hstack((init_rand, np.random.uniform(0, 2*np.pi, (N, 1))))
    
    inits = [init_fib, init_rand]
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(_energy_and_grad, init, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # If we have time, try more random restarts
    while time.time() < deadline - 10.0:
        init_rand = np.random.uniform(0, np.pi, (N, 1))
        init_rand = np.hstack((init_rand, np.random.uniform(0, 2*np.pi, (N, 1))))
        
        try:
            res = minimize(_energy_and_grad, init_rand, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback to Fibonacci
        best_params = init_fib
        
    # Convert back to Cartesian
    theta = best_params[:, 0]
    phi = best_params[:, 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack((x, y, z))
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts
