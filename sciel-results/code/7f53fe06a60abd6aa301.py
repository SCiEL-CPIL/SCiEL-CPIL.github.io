"""
Strategy: Basin Hopping with Riemannian Optimization.
The current program uses deterministic Riemannian CG from structured starts.
This program uses a stochastic search (Basin Hopping) where each step involves:
1. Randomly perturbing the configuration on the sphere (the 'kick').
2. Minimizing the energy using Riemannian L-BFGS-B (the 'relaxation').
This allows escaping local minima that trap pure gradient methods.
"""

N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff[i, j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    # We use a safe distance to avoid log(0) if points coincide, though 
    # the kick/relaxation loop should prevent this.
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    # This is the Euclidean gradient. We project it to the tangent space later.
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(x, grad_euclidean):
    """Project Euclidean gradient to tangent space."""
    # g_tangent = g_euclidean - (g_euclidean . x) x
    return grad_euclidean - np.sum(grad_euclidean * x, axis=1, keepdims=True) * x

def _exp_map_sphere(x, v):
    """
    Exponential map on unit sphere.
    x: (N, 3) current points
    v: (N, 3) tangent vectors
    Returns: (N, 3) new points on sphere
    """
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    safe_theta = np.where(theta < 1e-10, 1.0, theta)
    safe_v = np.where(theta < 1e-10, v, v / safe_theta)
    
    x_new = np.cos(theta) * x + np.sin(safe_theta) * safe_v
    
    # For zero vectors, just return x
    zero_mask = (theta < 1e-10)[:, 0]
    if np.any(zero_mask):
        x_new[zero_mask] = x[zero_mask]
        
    # Normalize for safety
    norms = np.linalg.norm(x_new, axis=1, keepdims=True)
    x_new = x_new / norms
    return x_new

def _log_map_sphere(x, y):
    """
    Logarithmic map on unit sphere: log_x(y).
    Returns v such that exp_x(v) = y.
    x, y: (N, 3)
    """
    cos_theta = np.sum(x * y, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Handle antipodal points (theta ~ pi) - unlikely in minimization but possible in kicks
    antipodal = theta > (np.pi - 1e-6)
    
    sin_theta = np.sin(theta)
    safe_sin = np.where(sin_theta < 1e-10, 1.0, sin_theta)
    
    # v = (theta / sin(theta)) * (y - cos(theta) * x)
    y_minus_cos_x = y - cos_theta[:, np.newaxis] * x
    v = (theta[:, np.newaxis] / safe_sin[:, np.newaxis]) * y_minus_cos_x
    
    # For points very close to x, v should be ~ y - x
    close_mask = (theta < 1e-6)
    if np.any(close_mask):
        v[close_mask] = y[close_mask] - x[close_mask]
        
    # For antipodal points, the log map is undefined; return a large vector 
    # or handle specially. In basin hopping, we shouldn't hit exact antipodes often.
    if np.any(antipodal):
        # Pick an arbitrary tangent direction perpendicular to x
        # This is a heuristic for a degenerate case
        v[antipodal] = 10.0 * x[antipodal] # Will be projected away in grad, but exp might fail
        # Better: just set to 0 or small random? Let's set to 0 to be safe, 
        # effectively saying "no movement" for those specific points in this step.
        v[antipodal] = 0.0

    return v

def _riemannian_l_bfgs(x0, max_iter=100, tol=1e-8):
    """
    Riemannian L-BFGS using scipy's L-BFGS-B in a retraction framework.
    We map the sphere to Euclidean space via tangent vectors, optimize, and map back.
    Actually, simpler: use scipy's L-BFGS-B on the flat coordinates, but project
    the gradient to the tangent space and use the exponential map for the step?
    
    A more robust approach for scipy: 
    Define a function that takes flat Euclidean coords, projects to sphere, 
    computes energy and gradient, and returns them. But this ignores the manifold structure 
    in the step size.
    
    Better: Implement a simple Riemannian L-BFGS manually or use 
    scipy.optimize.minimize with 'L-BFGS-B' on the tangent space coordinates 
    iteratively.
    
    Let's stick to a robust Riemannian Conjugate Gradient or a simplified L-BFGS 
    that projects gradients.
    
    For simplicity and speed, let's use a Riemannian Gradient Descent with adaptive 
    step size, or a Riemannian CG. Given the time budget, a well-tuned CG is safer 
    than implementing a full Riemannian L-BFGS from scratch which might have bugs.
    
    However, L-BFGS is much faster to converge. Let's try to use scipy's L-BFGS-B 
    on the "unwrapped" coordinates if we had a chart, but the sphere has no global chart.
    
    Alternative: Use scipy.optimize.minimize on the (N,3) array with bounds? No.
    
    Let's use a Riemannian Steepest Descent with Backtracking Line Search, 
    but accelerated with momentum (Riemannian Nesterov) or just CG.
    The current program uses CG. Let's try to make the CG more robust or 
    switch to a method that handles the multimodality better via the outer loop.
    
    Actually, the key is the OUTER loop (Basin Hopping). The inner optimizer 
    just needs to be good enough to find the local minimum. 
    Let's reuse the _riemannian_cg from the current program but ensure it's robust.
    """
    
    x = x0.copy()
    E, grad_eucl = _log_energy_and_grad_sphere(x)
    g = _riemannian_grad(x, grad_eucl)
    
    p = -g  # Initial search direction
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    p = np.where(p_norm < 1e-15, p, p / p_norm)
    
    best_energy = E
    best_x = x.copy()
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Line search
        alpha = 1.0
        for ls_iter in range(30):
            v = alpha * p
            x_trial = _exp_map_sphere(x, v)
            E_trial, grad_eucl_trial = _log_energy_and_grad_sphere(x_trial)
            
            if E_trial < E:
                break
            alpha *= 0.5
        
        if not (E_trial < E):
            alpha = 0.01
            v = alpha * p
            x_trial = _exp_map_sphere(x, v)
            E_trial, grad_eucl_trial = _log_energy_and_grad_sphere(x_trial)
        
        x = x_trial
        g_trial = _riemannian_grad(x, grad_eucl_trial)
        
        if E_trial < best_energy:
            best_energy = E_trial
            best_x = x.copy()
        
        # Parallel transport p
        p = p - np.sum(p * x, axis=1, keepdims=True) * x
        
        # Polak-Ribiere update
        g_diff = g_trial - g
        beta = max(0.0, np.sum(g_trial * g_diff) / (np.sum(g * g) + 1e-15))
        
        g = g_trial
        p = -g + beta * p
        E = E_trial
        
    return best_x, best_energy

def _kick(x, scale):
    """
    Apply a random kick to the configuration on the sphere.
    x: (N, 3)
    scale: magnitude of perturbation
    Returns: (N, 3)
    """
    # Generate random tangent vectors
    noise = np.random.randn(*x.shape)
    # Project to tangent space
    noise_tangent = noise - np.sum(noise * x, axis=1, keepdims=True) * x
    
    # Scale
    v = scale * noise_tangent
    
    # Apply exponential map
    x_new = _exp_map_sphere(x, v)
    return x_new

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initial configuration: Fibonacci sphere is a good start
    x_current = _fibonacci_sphere(N)
    x_current /= np.linalg.norm(x_current, axis=1, keepdims=True)
    
    # Initial relaxation
    x_current, E_current = _riemannian_l_bfgs(x_current, max_iter=50, tol=1e-6)
    if E_current < best_energy:
        best_energy = E_current
        best_x = x_current.copy()
        
    # Basin Hopping Loop
    kick_scale = 0.5  # Initial kick scale
    min_kick_scale = 0.01
    
    # We will run until time runs out
    # Adaptive kick scale: decrease if we find better solutions, increase if stuck?
    # Simple strategy: fixed schedule or annealing.
    # Let's use a simple annealing schedule for kick_scale.
    
    rng = np.random.default_rng(42)
    
    # Pre-compute some random tangent directions? No, just generate on the fly.
    
    # To speed up, we can skip the full CG relaxation for small kicks?
    # No, relaxation is necessary to find the local min.
    
    # Optimization: If the kick is very small, the local min is likely the same.
    # But we need to escape.
    
    # Let's monitor time per iteration to adjust max_iter in relaxation if needed.
    # For now, fixed max_iter=30 for speed, as we have many iterations.
    
    max_iter_relax = 30
    tol_relax = 1e-5
    
    last_improvement_time = start_time
    stagnation_count = 0
    
    while time.time() < deadline:
        # Generate kick
        x_kicked = _kick(x_current, kick_scale)
        
        # Relax
        x_relaxed, E_relaxed = _riemannian_l_bfgs(x_kicked, max_iter=max_iter_relax, tol=tol_relax)
        
        if E_relaxed < best_energy:
            best_energy = E_relaxed
            best_x = x_relaxed.copy()
            x_current = x_relaxed.copy()
            last_improvement_time = time.time()
            stagnation_count = 0
            # Optionally decrease kick scale slightly on success
            # kick_scale *= 0.95
        else:
            # Even if not better than global best, accept if better than current local min?
            # Standard Basin Hopping accepts if better than current x_current's energy.
            # But we always relax to local min. So E_relaxed is the local min energy.
            # If E_relaxed < E_current (energy of x_current before kick), we accept.
            # This is Metropolis acceptance at T=0 (greedy).
            # To explore more, we might want to accept worse solutions occasionally?
            # For minimization, greedy is often sufficient if kicks are large enough.
            # Let's stick to greedy: accept if it improves the *current* local minimum.
            if E_relaxed < E_current:
                x_current = x_relaxed.copy()
                E_current = E_relaxed
                stagnation_count = 0
            else:
                stagnation_count += 1
                # If stuck for a while, increase kick scale to jump out
                if stagnation_count > 5:
                    kick_scale = min(kick_scale * 1.5, 1.0)
                    stagnation_count = 0
                    # Also try a larger kick immediately?
                    # The loop will handle it.
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # Final polish with more iterations
    if time.time() < deadline - 2 and best_x is not None:
        try:
            x_final, E_final = _riemannian_l_bfgs(best_x, max_iter=100, tol=1e-8)
            if E_final < best_energy:
                best_x = x_final
                best_energy = E_final
        except Exception:
            pass

    if best_x is None:
        best_x = x_current
        best_energy = _log_energy_and_grad_sphere(best_x)[0]

    # Ensure non-zero and normalized
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_x = best_x / norms
    
    return best_x

# EVOLVE-BLOCK-END
