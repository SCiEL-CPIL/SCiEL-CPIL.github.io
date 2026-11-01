"""
Strategy: Stochastic Gradient Langevin Dynamics (SGLD) with Simulated Annealing.
1. Treat the minimization of logarithmic energy as a sampling problem from the 
   Boltzmann distribution P(x) ~ exp(-E(x)/T).
2. Use Langevin dynamics: x_{t+1} = x_t - eta * grad(E) + sqrt(2*eta*T) * noise.
3. On the sphere, use the exponential map to ensure points remain on the manifold.
4. Anneal temperature T from high to low over time. High T allows exploration 
   (escaping local minima), low T allows precise convergence.
5. This stochastic approach is robust to the ill-conditioning of the log-energy 
   Hessian near collisions, which often traps deterministic second-order methods.
"""

N = 150
import numpy as np
import time
import math

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_project(grad, x):
    """
    Project Euclidean gradient to the tangent space of the sphere at x.
    proj_x(g) = g - (g . x) x
    """
    dot = np.sum(grad * x, axis=1, keepdims=True)
    return grad - dot * x

def _exponential_map(x, v):
    """
    Exponential map on the unit sphere.
    x: point on sphere (N, 3)
    v: tangent vector (N, 3)
    Returns: new point on sphere.
    """
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    mask = norm_v > 1e-10
    new_x = np.zeros_like(x)
    
    cos_theta = np.cos(norm_v)
    sin_theta_over_norm = np.zeros_like(norm_v)
    sin_theta_over_norm[mask] = np.sin(norm_v[mask]) / norm_v[mask]
    
    new_x = cos_theta * x + sin_theta_over_norm * v
    return new_x

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _sgld_anneal(x_init, total_steps=2000, t0=1.0, t1=1e-6, eta0=0.01, eta1=0.001):
    """
    Run Stochastic Gradient Langevin Dynamics with linear annealing.
    """
    x = x_init.copy()
    current_energy, _ = _log_energy_and_grad_sphere(x)
    best_energy = current_energy
    best_x = x.copy()
    
    # Pre-compute log temperature schedule for efficiency if needed, 
    # but linear interpolation in log-space is good for annealing.
    log_t0 = np.log(t0)
    log_t1 = np.log(t1)
    log_eta0 = np.log(eta0)
    log_eta1 = np.log(eta1)
    
    rng = np.random.default_rng(42)
    
    for step in range(total_steps):
        # Annealing schedule: exponential decay of T and eta
        frac = step / max(1, total_steps - 1)
        log_T = log_t0 + (log_t1 - log_t0) * frac
        T = np.exp(log_T)
        
        log_eta = log_eta0 + (log_eta1 - log_eta0) * frac
        eta = np.exp(log_eta)
        
        # Compute gradient
        energy, grad = _log_energy_and_grad_sphere(x)
        
        # Project gradient to tangent space
        rgrad = _riemannian_project(grad, x)
        
        # Langevin step:
        # Deterministic part: -eta * rgrad
        # Stochastic part: sqrt(2 * eta * T) * noise
        # Noise must be in tangent space. 
        # Generate random vectors and project them.
        noise = rng.normal(size=x.shape)
        noise = _riemannian_project(noise, x)
        
        step_size = eta
        noise_scale = np.sqrt(2.0 * eta * T)
        
        v = -step_size * rgrad + noise_scale * noise
        
        # Move using exponential map
        x_new = _exponential_map(x, v)
        
        # Evaluate new energy
        e_new, _ = _log_energy_and_grad_sphere(x_new)
        
        if e_new < best_energy:
            best_energy = e_new
            best_x = x_new.copy()
        
        # Accept or reject? In SGLD for optimization, we usually just take the step 
        # because the noise helps escape. However, if the step is too large and increases 
        # energy significantly at low T, we might want to damp. 
        # Standard SGLD doesn't have Metropolis accept/reject, but for stability at low T,
        # we can check if the step is "valid".
        # If e_new is inf or nan, revert.
        if not np.isfinite(e_new):
            x = best_x.copy()
            continue
            
        x = x_new
        
    return best_x, best_energy

def _polish_newton(x_init, max_iter=100, tol=1e-10):
    """
    A simple Riemannian Newton-like polish using gradient descent with adaptive step size
    for the final stage, as SGLD might leave points slightly off the precise minimum.
    """
    x = x_init.copy()
    energy, grad = _log_energy_and_grad_sphere(x)
    rgrad = _riemannian_project(grad, x)
    
    step_size = 0.1
    for i in range(max_iter):
        if np.linalg.norm(rgrad) < tol:
            break
            
        # Try to take a step
        # Use a simple line search
        alpha = step_size
        improved = False
        for _ in range(10):
            v = -alpha * rgrad
            x_new = _exponential_map(x, v)
            e_new, grad_new = _log_energy_and_grad_sphere(x_new)
            if e_new < energy:
                x = x_new
                energy = e_new
                rgrad = _riemannian_project(grad_new, x)
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-8:
                break
        else:
            step_size = min(step_size * 1.1, 0.5) # Slowly increase step size if successful
            
    return x, energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    # Use multiple diverse inits
    inits = [
        _fibonacci_sphere(N),
    ]
    
    # Add a few random inits for diversity
    rng = np.random.default_rng(123)
    for _ in range(3):
        pts = rng.normal(size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    best_energy = float('inf')
    best_x = None
    
    # Run SGLD Annealing on each initialization
    # We need to budget steps carefully. 
    # 150s total. 
    # Let's say 4 inits. 
    # If we run 1000 steps each, that's 4000 energy evals.
    # Each eval is O(N^2) = 22500 ops. 4000 * 22500 = 90M ops. Should be fast in NumPy.
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        # Run SGLD
        x_opt, e_opt = _sgld_anneal(x_init, total_steps=1500)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt.copy()
            
    # Polish the best result found
    if best_x is not None and time.time() < deadline - 2:
        x_final, e_final = _polish_newton(best_x)
        if e_final < best_energy:
            best_energy = e_final
            best_x = x_final
            
    return best_x

# EVOLVE-BLOCK-END
