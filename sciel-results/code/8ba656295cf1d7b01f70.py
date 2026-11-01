"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, basinhopping
import time

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci lattice."""
    points = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        points[i, 0] = np.cos(theta) * radius
        points[i, 1] = y
        points[i, 2] = np.sin(theta) * radius
    return points

def _generate_probe_points(num_probes=2000):
    """Generate a fixed set of probe points on the sphere for covering radius estimation."""
    # Use a denser Fibonacci sphere for probes
    return _fibonacci_sphere(num_probes)

def _smooth_covering_radius_and_grad(x_flat, probes, beta=100.0, lambda_pen=100.0):
    """
    Compute the smoothed covering radius (LogSumExp of min distances) 
    and its gradient, plus a penalty for non-unit norm.
    """
    x = x_flat.reshape(-1, 3)
    # Normalize x to be on the sphere for distance calculations
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x_norm = x / norms
    
    # Compute distances from each probe to all points
    # probes: (M, 3), x_norm: (N, 3)
    # dot: (M, N)
    dots = np.dot(probes, x_norm.T)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # (M, N)
    
    # Find min distance for each probe
    min_dists = np.min(dists, axis=1) # (M,)
    min_indices = np.argmin(dists, axis=1) # (M,)
    
    # LogSumExp for stability
    max_d = np.max(min_dists)
    exp_terms = np.exp(beta * (min_dists - max_d))
    sum_exp = np.sum(exp_terms)
    log_sum_exp = max_d + np.log(sum_exp) / beta
    
    # Penalty term for normalization
    pen_val = lambda_pen * np.sum((np.sum(x**2, axis=1) - 1.0)**2)
    total_val = log_sum_exp + pen_val
    
    # Gradient
    # Gradient of LogSumExp:
    # w_p = exp(beta * (d_p - max_d)) / sum_exp
    # grad_L = sum_p w_p * grad_{x_j} d_p
    # d_p = arccos(probe_p . x_j) where j = min_indices[p]
    # grad_{x_j} d_p = - (probe_p - (probe_p . x_j) x_j) / sin(d_p)
    # Note: This gradient is with respect to x_j on the sphere.
    # We need to project it or just use it? 
    # Since we are adding a penalty for ||x||!=1, the raw gradient in R3 is okay, 
    # but the "sphere gradient" is more accurate.
    # Let's compute the raw gradient contributions.
    
    grad = np.zeros_like(x)
    
    # Weights for each probe
    weights = exp_terms / sum_exp # (M,)
    
    # For each probe p, we need to add contribution to x[min_indices[p]]
    # Vectorized approach:
    # We have weights (M,), min_indices (M,)
    # We need to compute grad_d for each (p, min_indices[p])
    
    # Get the specific dots for the min indices
    # dots_min = dots[np.arange(M), min_indices]
    # sin_d = np.sin(min_dists)
    # proj = probes - dots_min[:, None] * x_norm[min_indices]
    # grad_vecs = -proj / sin_d[:, None]
    
    M = len(probes)
    idx = np.arange(M)
    dots_min = dots[idx, min_indices]
    sin_d = np.sin(min_dists)
    # Avoid division by zero
    sin_d[sin_d < 1e-12] = 1e-12
    
    # proj = probes - dots_min * x_norm[min_indices]
    proj = probes - dots_min[:, None] * x_norm[min_indices]
    grad_vecs = -proj / sin_d[:, None] # (M, 3)
    
    # Accumulate gradient
    # For each p, add weights[p] * grad_vecs[p] to grad[min_indices[p]]
    # Use np.add.at for unbuffered accumulation
    np.add.at(grad, min_indices, weights[:, None] * grad_vecs)
    
    # Add penalty gradient
    # pen = lambda * sum (||x_i||^2 - 1)^2
    # d pen / d x_i = lambda * 2 * (||x_i||^2 - 1) * 2 * x_i = 4 * lambda * (||x_i||^2 - 1) * x_i
    pen_grad = 4.0 * lambda_pen * (np.sum(x**2, axis=1, keepdims=True) - 1.0) * x
    
    total_grad = grad + pen_grad
    
    return total_val, total_grad.flatten()

def construct():
    # Initial guess
    x0 = _fibonacci_sphere(N)
    x0_flat = x0.flatten()
    
    # Generate probes
    probes = _generate_probe_points(2000)
    
    # Define objective and jac for L-BFGS-B
    def fun(x):
        val, _ = _smooth_covering_radius_and_grad(x, probes)
        return val
    
    def jac(x):
        _, g = _smooth_covering_radius_and_grad(x, probes)
        return g
    
    # Local minimization
    # We use L-BFGS-B. 
    # Note: The points might drift off the sphere, but the penalty keeps them close.
    # We should normalize after optimization.
    
    result = minimize(fun, x0_flat, jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    
    x_opt = result.x.reshape(-1, 3)
    # Normalize
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    # Basin Hopping for global improvement
    # We wrap the L-BFGS-B in a basinhopping call
    
    def bh_fun(x):
        # Normalize input for consistency in perturbation? 
        # Basinhopping perturbs in the parameter space.
        # We want to perturb on the sphere.
        # Standard basinhopping perturbs x -> x + delta.
        # If we normalize x in the objective, the landscape is smooth enough.
        val, _ = _smooth_covering_radius_and_grad(x, probes)
        return val
        
    def bh_jac(x):
        _, g = _smooth_covering_radius_and_grad(x, probes)
        return g

    # Run Basin Hopping
    # stepsize=0.1 is a reasonable perturbation
    # niter=50 to keep time down
    bh_result = basinhopping(bh_fun, x_opt.flatten(), 
                             jac=bh_jac, 
                             stepsize=0.1, 
                             niter=30, 
                             minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 50}})
    
    x_final = bh_result.x.reshape(-1, 3)
    x_final = x_final / np.linalg.norm(x_final, axis=1, keepdims=True)
    
    return x_final
# EVOLVE-BLOCK-END
