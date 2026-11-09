"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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

def _energy_and_grad(pts):
    """
    Vectorized computation of logarithmic energy and gradient on the sphere.
    pts: (N, 3) array of points on the unit sphere.
    Returns: (energy, grad) where grad is the projected gradient in R^{N*3}.
    """
    # Compute pairwise squared distances
    # dist_sq[i, j] = |pts[i] - pts[j]|^2
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid log(0) by clipping
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i<j} log(d_sq)
    # S = sum_{i != j} log(d_sq_ij). Then sum_{i<j} log(d_sq_ij) = S/2.
    # E = -0.5 * (S/2) = -0.25 * S.
    
    # Gradient computation:
    # dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    w = inv_dist_sq
    sum_w = np.sum(w, axis=1)
    
    term1 = (pts.T @ w).T
    term2 = pts * sum_w[:, None]
    
    grad_raw = term1 - term2
    
    # Project gradient to tangent space of the sphere
    dot = np.sum(grad_raw * pts, axis=1)
    grad_tan = grad_raw - dot[:, None] * pts
    
    # Compute energy
    log_dist_sq = np.log(dist_sq)
    # Mask diagonal to 0 to avoid contribution from self
    log_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, log_dist_sq)
    
    energy = -0.25 * np.sum(log_dist_sq)
    
    return energy, grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _optimize_lbfgs(pts, deadline):
    """
    Optimize using L-BFGS-B with analytical gradient.
    We minimize the energy with respect to the flattened coordinates.
    Since the points must remain on the sphere, we use a penalty or projection.
    However, L-BFGS-B works on Euclidean space. 
    A common trick is to optimize on the sphere using the fact that the gradient 
    is already projected to the tangent space. But standard minimize expects 
    a function f: R^n -> R and gradient g: R^n -> R^n.
    
    If we pass the projected gradient, the step x_new = x - alpha * g_tan 
    will move off the sphere. 
    To handle this, we can define a function that projects the input to the sphere 
    before evaluation. But this makes the gradient discontinuous if not careful.
    
    Alternative: Use the fact that for small steps, the change in energy is 
    well-approximated by the tangent gradient. 
    However, scipy's L-BFGS-B does not support manifold constraints directly.
    
    Let's use a different approach: 
    Minimize the energy on the sphere by treating the coordinates as free 
    and adding a large penalty for deviation from the sphere, OR 
    simply re-project after each step in a custom callback? 
    Scipy doesn't support that easily.
    
    Better: Use the "box" constraint? No.
    
    Let's stick to the previous strategy's success: 
    The previous program used a custom CG. 
    But inspiration 1 used L-BFGS and got 1.0005. 
    How did it handle the sphere?
    Usually, one minimizes the energy function defined on R^{3N} where the 
    function internally normalizes the points. 
    f(x) = Energy(normalize(x)).
    The gradient of this function with respect to x is NOT simply the tangent 
    gradient. It involves the derivative of the normalization.
    
    Let's derive the gradient for f(x) = E(x/||x||).
    Let u = x/||x||.
    dE/du is the tangent gradient g_tan.
    dx/du = ...
    Actually, it is easier to just use the custom CG or a custom L-BFGS 
    that projects.
    
    However, there is a standard trick: 
    Minimize E(x) subject to ||x_i|| = 1.
    Using Lagrange multipliers, the gradient in R^3 is g_tan + lambda * x_i.
    For optimization, we can just use g_tan and project the step back to the sphere.
    
    Since scipy.minimize doesn't allow custom step projection, 
    we will implement a simple custom L-BFGS loop or use the existing 
    _conjugate_gradient from the parent which worked reasonably well (1.0005).
    
    Wait, the parent program's score is 1.0005. 
    The inspirations also scored 1.0005.
    The goal is to beat 1.0005.
    
    Let's try a multi-restart strategy with the existing CG, but with 
    better initializations and more restarts.
    The current program does 20 cycles of CG+SA. 
    Maybe we can do many pure CG runs from different starts.
    """
    # We will use the _conjugate_gradient defined below
    return _conjugate_gradient(pts, max_iter=1000, deadline=deadline)

def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    """
    Conjugate Gradient method on the sphere.
    """
    x = pts.copy()
    energy, grad = _energy_and_grad(x)
    p = -grad
    
    # Project p to tangent space
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    # Store previous gradient for Polak-Ribiere
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Line search: minimize f(x + alpha * p)
        # Use backtracking line search
        alpha = 0.1
        c1 = 1e-4
        rho = 0.5
        
        dir_deriv = np.sum(grad * p)
        
        # If direction is not a descent direction, restart
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
            
        found_step = False
        for _ in range(50):
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_grad(x_new)
            
            # Armijo condition
            if e_new <= energy + c1 * alpha * dir_deriv:
                found_step = True
                break
            alpha *= rho
            
        if not found_step:
            # If line search fails, try a smaller step or break
            break
        
        # Update
        x = x_new
        energy = e_new
        grad = g_new
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Compute beta (Polak-Ribiere)
        # beta = grad^T (grad - grad_prev) / (grad_prev^T grad_prev)
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        # New direction
        p_new = -grad + beta * p
        
        # Project p_new to tangent space
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        # Check if p_new is a descent direction
        if np.sum(grad * p_new) > 0:
            beta = 0.0 # Restart
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x

def construct():
    """
    Main entry point.
    Strategy:
    1. Generate multiple diverse initial configurations.
    2. Run CG optimization on each within a time budget.
    3. Return the best configuration found.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # List of initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    inits.append(_initial_fibonacci())
    
    # 2. Random points
    rng = np.random.default_rng(42)
    for i in range(10):
        pts = rng.normal(size=(N, 3))
        inits.append(_project_to_sphere(pts))
        
    # 3. Jittered Fibonacci
    for i in range(10):
        pts = _initial_fibonacci() + rng.normal(size=(N, 3)) * 0.1
        inits.append(_project_to_sphere(pts))
        
    # 4. Grid-like (spherical t-design approximation)
    # We can use a simple grid on the sphere
    # But random and Fibonacci are usually good.
    
    best_pts = inits[0].copy()
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Optimize each initial configuration
    # We have ~145s. Let's give each init ~5s.
    # 21 inits * 5s = 105s. Safe.
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 5:
            break
            
        # Calculate time budget for this run
        time_left = deadline - time.time()
        run_deadline = time.time() + min(5.0, time_left - 2.0)
        
        # If time is too tight, skip
        if run_deadline <= time.time():
            continue
            
        optimized_pts = _conjugate_gradient(init_pts, max_iter=500, deadline=run_deadline)
        
        e_opt, _ = _energy_and_grad(optimized_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = optimized_pts.copy()
            
    # Final polish on the best found
    if time.time() < deadline - 5:
        final_pts = _conjugate_gradient(best_pts, max_iter=2000, deadline=deadline - 1)
        e_final, _ = _energy_and_grad(final_pts)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = final_pts.copy()
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts
# EVOLVE-BLOCK-END
