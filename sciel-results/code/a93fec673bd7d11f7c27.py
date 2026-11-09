"""
Minimal logarithmic energy of 100 points on the sphere.
Combines CG-based local optimization with diverse initializations and
a high-precision L-BFGS-B polish pass.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere.
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
    """
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Clip to avoid log(0) and numerical issues
    dist_sq = np.maximum(dist_sq, 1e-12)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    log_dist_sq = np.log(dist_sq)
    # Set diagonal to 0 to ignore self-interaction
    np.fill_diagonal(log_dist_sq, 0.0)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient of E with respect to x_i:
    # E = -0.5 * sum_{i!=j} log(|x_i - x_j|)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / (2 * |x_i - x_j|^2)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    term1 = inv_dist_sq @ pts
    term2 = pts * np.sum(inv_dist_sq, axis=1)[:, None]
    
    grad_raw = 0.5 * (term1 - term2)
    
    # Project to tangent space
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    return energy, grad_tan

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _energy_and_grad_flat(x_flat):
    """
    Objective and gradient for scipy L-BFGS-B.
    x_flat: (3N,) vector.
    """
    x = x_flat.reshape(N, 3)
    x = _project_to_sphere(x)
    energy, grad_tan = _energy_and_tangent_grad(x)
    return energy, grad_tan.flatten()

def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(rng, scale=1.0):
    pts = rng.normal(size=(N, 3)) * scale
    return _project_to_sphere(pts)

def _initial_perturbed_fibonacci(rng, scale=0.1):
    pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * scale
    return _project_to_sphere(pts)

def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    """
    Conjugate Gradient method on the sphere with Armijo line search.
    """
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    
    # Project p to tangent space
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Line search
        alpha = 0.1
        c1 = 1e-4
        rho = 0.5
        
        dir_deriv = np.sum(grad * p)
        
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
        
        found_step = False
        for _ in range(50):
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            
            if e_new <= energy + c1 * alpha * dir_deriv:
                found_step = True
                break
            alpha *= rho
        
        if not found_step:
            break
        
        x = x_new
        energy = e_new
        grad = g_new
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Polak-Ribiere
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        p_new = -grad + beta * p
        
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        if np.sum(grad * p_new) > 0:
            beta = 0.0
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x

def _lbfgs_polish(pts, deadline):
    """
    High-precision L-BFGS-B polish on the best candidate.
    Uses very tight tolerances and many iterations.
    """
    x0 = pts.flatten()
    time_left = deadline - time.time()
    if time_left < 1.0:
        return pts
    
    try:
        res = minimize(
            _energy_and_grad_flat,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'ftol': 1e-15,
                'gtol': 1e-14,
                'maxls': 50,
                'maxfun': 10000
            }
        )
        if res.success or res.fun < _energy_and_tangent_grad(pts)[0]:
            return _project_to_sphere(res.x.reshape(N, 3))
    except Exception:
        pass
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s buffer before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_initial_fibonacci())
    
    # Random initializations with different scales
    for scale in [0.5, 1.0, 2.0]:
        inits.append(_initial_random(rng, scale=scale))
    
    # Perturbed Fibonacci with different perturbation scales
    for scale in [0.05, 0.1, 0.2, 0.3]:
        inits.append(_initial_perturbed_fibonacci(rng, scale=scale))
    
    # Additional random starts
    for _ in range(4):
        inits.append(_initial_random(rng, scale=1.0))
    
    best_pts = inits[0].copy()
    best_energy = _energy_and_tangent_grad(best_pts)[0]
    
    # Phase 1: Optimize each initialization with CG
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 25.0:
            break
        
        time_left = deadline - time.time()
        run_deadline = time.time() + min(4.0, time_left - 1.0)
        
        if run_deadline <= time.time():
            continue
        
        optimized_pts = _conjugate_gradient(init_pts, max_iter=800, deadline=run_deadline)
        
        e_opt, _ = _energy_and_tangent_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
    
    # Phase 2: L-BFGS-B polish on the best candidate
    if time.time() < deadline - 5.0:
        polished_pts = _lbfgs_polish(best_pts, deadline - 1.0)
        e_polish, _ = _energy_and_tangent_grad(polished_pts)
        
        if e_polish < best_energy:
            best_energy = e_polish
            best_pts = polished_pts.copy()
    
    # Phase 3: Final CG polish with tighter tolerance
    if time.time() < deadline - 2.0:
        final_pts = _conjugate_gradient(best_pts, max_iter=2000, tol=1e-12, deadline=deadline - 0.5)
        e_final, _ = _energy_and_tangent_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    best_pts = _project_to_sphere(best_pts)
    
    # Safety check: ensure no points are too close
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        best_pts = _initial_fibonacci()
    
    # Verify all values are finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
