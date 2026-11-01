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
    Energy: sum_{i<j} log(1/||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    Gradient: - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    Note: We compute for the full matrix and subtract diagonal/adjust for double counting if needed,
    but standard gradient for sum_{i<j} f(d_ij) is 0.5 * sum_{k!=i} grad_i f(d_ik) * 2? 
    Let's derive carefully.
    E = sum_{i<j} -log(d_ij)
    d_ij = ||x_i - x_j||
    dE/dx_i = sum_{j != i} dE/dx_i (only terms where i is involved)
    For term (i,j) with i<j: d/dx_i (-log d_ij) = - (1/d_ij) * (1/2 d_ij^2)^(-1/2) * 2(x_i - x_j) = -(x_i - x_j)/d_ij^2
    For term (k,i) with k<i: d/dx_i (-log d_ki) = - (x_i - x_k)/d_ki^2
    So grad_i E = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    # x is (N, 3)
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist is (N, N)
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    # We only care about i != j
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Energy
    # sum_{i<j} log(1/dist) = 0.5 * sum_{i != j} log(1/dist)
    # log(1/d) = -log(d)
    # We compute sum over all i!=j of -log(dist), then divide by 2.
    # To avoid log(0), we set diagonal to 1.0 before log, but mask it out in sum.
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient
    # grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # inv_dist_sq is (N, N)
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # weighted_diff[i,j] = (x_i - x_j) / dist_ij^2
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    # grad[i] = - sum_{j} weighted_diff[i,j]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _project_to_sphere(x):
    """Normalizes rows of x to unit length."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return x / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_x = x0
    best_energy, _ = _log_energy_and_grad(x0)
    
    # We will use L-BFGS-B which works on R^3. 
    # However, L-BFGS-B assumes Euclidean space. 
    # To handle the sphere constraint, we can either:
    # 1. Use Lagrange multipliers (complex)
    # 2. Parameterize the sphere (e.g., spherical coordinates - bad near poles)
    # 3. Use a projection method: minimize f(x) on R^3, but project back to sphere after each step? 
    #    Standard trick: Minimize f(x) subject to ||x_i||=1.
    #    If we use L-BFGS-B on the unconstrained x, it will drift off the sphere.
    #    Better: Use the fact that the gradient on the sphere is the projection of the Euclidean gradient.
    #    grad_sphere_i = grad_euclidean_i - (grad_euclidean_i . x_i) x_i
    #    Since x_i is unit vector.
    #    However, L-BFGS-B doesn't natively support this projection in the line search.
    #    
    # Alternative: Use scipy.optimize.minimize with method='L-BFGS-B' but define the function
    # to take x, project it to sphere, compute energy, and return projected gradient?
    # No, the optimizer steps in the parameter space. If parameter space is R^3, it doesn't know about sphere.
    
    # Let's try a different approach: Use `scipy.optimize.minimize` with `method='L-BFGS-B'`
    # but we need to handle the constraint.
    # Actually, a common and effective heuristic for sphere problems is to just run L-BFGS-B on the 3N coordinates
    # and periodically re-normalize, or use a trust-region method.
    # 
    # However, there is a simpler, more robust method for this specific task:
    # Use `scipy.optimize.minimize` with `method='L-BFGS-B'` on the **spherical coordinates**? No, singularities.
    # 
    # Let's use the **projection gradient descent** approach manually, or better, use `scipy.optimize.minimize`
    # with `method='L-BFGS-B'` but we must provide the gradient correctly.
    # If we define the objective as E(x) where x is in R^{3N}, and we project x to sphere inside the function,
    # the gradient returned must be consistent with the projection.
    # 
    # Actually, the most reliable standard library/scipy way without custom solvers is to use
    # `scipy.optimize.minimize` with `method='L-BFGS-B'` on the **unconstrained** variables, but this is tricky.
    
    # Let's use a well-known trick: 
    # The gradient of the energy with respect to the points on the sphere is the tangential component of the Euclidean gradient.
    # If we use a generic optimizer that supports constraints, like SLSQP, it might be slow.
    # 
    # Let's try `scipy.optimize.minimize` with `method='L-BFGS-B'` but we will use a **retraction** based approach.
    # Specifically, we will optimize on R^3 but after each function evaluation, we project.
    # BUT, the gradient of the composed function f(project(x)) is J_p^T grad_f(p).
    # This is complex to implement correctly for L-BFGS-B.
    
    # Instead, let's use a simple, robust **Gradient Descent with Momentum** on the sphere,
    # implemented with vectorized NumPy operations. This is often sufficient for N=150 if we have enough steps.
    # Or, we can use `scipy.optimize.minimize` with `method='L-BFGS-B'` on the **2N** parameters (theta, phi) for each point?
    # That has singularities at poles.
    
    # Let's stick to the **Projected Gradient Descent** (or just Gradient Descent on the manifold).
    # Algorithm:
    # 1. Compute Euclidean gradient g.
    # 2. Project g to tangent space: g_tan = g - (g . x) x.
    # 3. Update: x_new = x - alpha * g_tan.
    # 4. Normalize x_new to project back to sphere.
    # 5. Adaptive step size.
    
    # This is safe and fast with vectorization.
    
    def step(x, alpha):
        energy, grad = _log_energy_and_grad(x)
        # Project gradient to tangent space
        # g_tan_i = grad_i - (grad_i . x_i) x_i
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_tan = grad - dot * x
        x_new = x - alpha * grad_tan
        x_new = _project_to_sphere(x_new)
        return x_new, energy, grad_tan
    
    # Adaptive step size
    alpha = 1e-3
    x_curr = x0.copy()
    
    # Run optimization
    while time.time() < deadline:
        x_new, e_curr, _ = step(x_curr, alpha)
        
        # Check if energy improved
        e_new, _ = _log_energy_and_grad(x_new)
        
        if e_new < e_curr:
            x_curr = x_new
            e_curr = e_new
            # Slightly increase alpha if improving
            alpha = min(alpha * 1.01, 1e-2)
        else:
            # Decrease alpha
            alpha *= 0.9
            # If alpha is too small, break or try restart?
            if alpha < 1e-8:
                break
            
        # Update best
        if e_curr < best_energy:
            best_energy = e_curr
            best_x = x_curr.copy()
            
        # Check for convergence (small gradient norm)
        # We already computed grad in step, but let's check norm of grad_tan
        # Actually, step returns grad_tan. We didn't store it. Let's recompute or modify step.
        # For efficiency, we can just check energy change.
        
        # Simple convergence: if change is very small
        # We can also check if we are stuck.
        
        # Let's add a simple check: if alpha is very small, try a random perturbation?
        # For now, just break if alpha is tiny.
        
    return best_x

# EVOLVE-BLOCK-END
