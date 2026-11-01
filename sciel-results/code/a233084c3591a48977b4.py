"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _riemannian_grad(x, grad_euclidean):
    """Project Euclidean gradient to tangent space."""
    return grad_euclidean - np.sum(grad_euclidean * x, axis=1, keepdims=True) * x

def _exp_map_sphere(x, v):
    """Exponential map on unit sphere."""
    theta = np.linalg.norm(v, axis=1, keepdims=True)
    # Avoid division by zero
    safe_theta = np.where(theta < 1e-10, 1.0, theta)
    safe_v = np.where(theta < 1e-10, v, v / safe_theta)
    x_new = np.cos(theta) * x + np.sin(safe_theta) * safe_v
    # For zero vectors, just return x
    zero_mask = (theta < 1e-10)[:, 0]
    x_new[zero_mask] = x[zero_mask]
    # Normalize for safety
    x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
    return x_new

def _log_map_sphere(x, y):
    """Logarithmic map on unit sphere: log_x(y)."""
    # Returns v such that exp_x(v) = y
    cos_theta = np.sum(x * y, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    # Handle antipodal points (theta ~ pi)
    antipodal = theta > (np.pi - 1e-6)
    if np.any(antipodal):
        # For antipodal points, choose an arbitrary tangent direction
        # This is a degenerate case; just return a small vector
        pass
    
    sin_theta = np.sin(theta)
    safe_sin = np.where(sin_theta < 1e-10, 1.0, sin_theta)
    
    # v = (theta / sin(theta)) * (y - cos(theta) * x)
    y_minus_cos_x = y - cos_theta[:, np.newaxis] * x
    v = (theta[:, np.newaxis] / safe_sin[:, np.newaxis]) * y_minus_cos_x
    
    # For points very close to x, v should be ~ y - x
    close_mask = (theta < 1e-6)
    v[close_mask] = y[close_mask] - x[close_mask]
    
    # For antipodal points, the log map is undefined; return a large vector
    v[antipodal] = 10.0 * x[antipodal]  # Arbitrary, but will fail in exp
    
    return v

def _riemannian_cg(x0, max_iter=200, tol=1e-10):
    """Riemannian Conjugate Gradient method for minimizing energy on the sphere."""
    x = x0.copy()
    E, grad_eucl = _log_energy_and_grad_sphere(x)
    g = _riemannian_grad(x, grad_eucl)
    
    p = -g  # Initial search direction
    # Normalize p
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    p = np.where(p_norm < 1e-15, p, p / p_norm)
    
    best_energy = E
    best_x = x.copy()
    
    for k in range(max_iter):
        # Check convergence
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Line search: find step size alpha
        alpha = 1.0
        for ls_iter in range(30):
            # Step in tangent space
            v = alpha * p
            x_trial = _exp_map_sphere(x, v)
            
            E_trial, grad_eucl_trial = _log_energy_and_grad_sphere(x_trial)
            
            if E_trial < E:
                break
            alpha *= 0.5
        
        if not (E_trial < E):
            # Try a smaller step
            alpha = 0.01
            v = alpha * p
            x_trial = _exp_map_sphere(x, v)
            E_trial, grad_eucl_trial = _log_energy_and_grad_sphere(x_trial)
        
        # Update
        x = x_trial
        g_trial = _riemannian_grad(x, grad_eucl_trial)
        
        if E_trial < best_energy:
            best_energy = E_trial
            best_x = x.copy()
        
        # Parallel transport p to new tangent space
        # PT_{x->x_trial}(p) = p - (p . (x - x_trial)/|x - x_trial|^2) * (x - x_trial)
        # More robust: use the Riemannian gradient change
        # Simple parallel transport along the geodesic
        delta_x = x - x0 if k == 0 else x - x  # This is 0, need previous x
        
        # Let's use a simpler parallel transport:
        # The tangent space is T_x S^2. A vector v in T_x S^2 is transported to T_{x_new} S^2 by:
        # v_new = v - (v . x_new) x_new
        p = p - np.sum(p * x, axis=1, keepdims=True) * x
        
        # Polak-Ribiere update
        g_diff = g_trial - g
        rho = np.sum(p * g_diff)
        beta = max(0.0, np.sum(g_trial * g_diff) / (np.sum(g * g) + 1e-15))
        
        g = g_trial
        p = -g + beta * p
        E = E_trial
        
    return best_x, best_energy

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _halton_sphere(n, base1=2, base2=3):
    """Generate a Halton sequence on the sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    i = np.arange(1, n + 1)
    u = np.array([halton(idx, base1) for idx in i])
    v = np.array([halton(idx, base2) for idx in i])
    
    # Map to sphere using uniform sampling on sphere
    # z = 1 - 2u
    # theta = 2*pi*v
    z = 1.0 - 2.0 * u
    theta = 2.0 * np.pi * v
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return x

def _random_sphere(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _so3_rotation_vectorize(R):
    """Apply a 3x3 rotation matrix to an (N,3) array."""
    return x @ R.T if hasattr(R, 'T') else x @ np.array(R).T

def _so3_rotate(x, R):
    """Apply rotation matrix R to points x."""
    return x @ R.T

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Generate multiple diverse initial configurations
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N, 2, 3),
        _halton_sphere(N, 3, 5),
        _random_sphere(N, 42),
        _random_sphere(N, 123),
    ]
    
    # Normalize
    for i in range(len(inits)):
        inits[i] /= np.linalg.norm(inits[i], axis=1, keepdims=True)
    
    # For each initial configuration, run Riemannian CG
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        try:
            x_polished, E_polished = _riemannian_cg(x_init, max_iter=100, tol=1e-10)
            
            if E_polished < best_energy:
                best_energy = E_polished
                best_x = x_polished.copy()
                
        except Exception:
            # Fallback: just use the initial configuration
            E_init, _ = _log_energy_and_grad_sphere(x_init)
            if E_init < best_energy:
                best_energy = E_init
                best_x = x_init.copy()
    
    # If no valid result, return the first init
    if best_x is None:
        best_x = inits[0]
        best_energy = _log_energy_and_grad_sphere(best_x)[0]
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 5:
        try:
            x_final, E_final = _riemannian_cg(best_x, max_iter=200, tol=1e-12)
            if E_final < best_energy:
                best_x = x_final
                best_energy = E_final
        except Exception:
            pass
    
    # Ensure non-zero and normalized
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    best_x = best_x / norms
    
    return best_x

# EVOLVE-BLOCK-END
