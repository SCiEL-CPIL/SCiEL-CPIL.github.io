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
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
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
    safe_theta = np.where(theta < 1e-10, 1.0, theta)
    safe_v = np.where(theta < 1e-10, v, v / safe_theta)
    x_new = np.cos(theta) * x + np.sin(safe_theta) * safe_v
    zero_mask = (theta < 1e-10)[:, 0]
    x_new[zero_mask] = x[zero_mask]
    x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)
    return x_new

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
    
    z = 1.0 - 2.0 * u
    theta = 2.0 * np.pi * v
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    return x

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _random_sphere(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _tangent_space_optimize(x0, max_iter=300, tol=1e-10):
    """
    Optimize in tangent space using L-BFGS-B on the flattened tangent vectors.
    We parameterize each point as x_i + v_i where v_i is in T_{x_i} S^2,
    then use exp map to get back to sphere.
    """
    def objective_and_grad(v_flat):
        v = v_flat.reshape(N, 3)
        # Project v to tangent space at x0
        v_proj = v - np.sum(v * x0, axis=1, keepdims=True) * x0
        x_new = _exp_map_sphere(x0, v_proj)
        E, grad_eucl = _log_energy_and_grad_sphere(x_new)
        # Pull back gradient: dE/dv = grad_eucl projected to tangent at x0
        # Actually, for exp map, the gradient in v-space is approximately
        # the Euclidean gradient of the energy evaluated at x_new, projected
        # back to the original tangent space via parallel transport approximation.
        # For small steps, grad_v ≈ grad_eucl (since x_new ≈ x0)
        grad_v = grad_eucl - np.sum(grad_eucl * x0, axis=1, keepdims=True) * x0
        return E, grad_v.flatten()
    
    v0 = np.zeros(N * 3)
    result = minimize(
        lambda v: objective_and_grad(v)[0],
        v0,
        jac=lambda v: objective_and_grad(v)[1],
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 20}
    )
    
    v_opt = result.x.reshape(N, 3)
    v_opt = v_opt - np.sum(v_opt * x0, axis=1, keepdims=True) * x0
    x_opt = _exp_map_sphere(x0, v_opt)
    E_opt, _ = _log_energy_and_grad_sphere(x_opt)
    return x_opt, E_opt

def _riemannian_cg(x0, max_iter=200, tol=1e-10):
    """Riemannian Conjugate Gradient method for minimizing energy on the sphere."""
    x = x0.copy()
    E, grad_eucl = _log_energy_and_grad_sphere(x)
    g = _riemannian_grad(x, grad_eucl)
    
    p = -g
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    p = np.where(p_norm < 1e-15, p, p / p_norm)
    
    best_energy = E
    best_x = x.copy()
    
    for k in range(max_iter):
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        alpha = 1.0
        E_trial = E
        x_trial = x
        grad_eucl_trial = grad_eucl
        for ls_iter in range(30):
            v = alpha * p
            x_trial = _exp_map_sphere(x, v)
            E_trial, grad_eucl_trial = _log_energy_and_grad_sphere(x_trial)
            if E_trial < E:
                break
            alpha *= 0.5
        
        x = x_trial
        g_trial = _riemannian_grad(x, grad_eucl_trial)
        
        if E_trial < best_energy:
            best_energy = E_trial
            best_x = x.copy()
        
        # Parallel transport p to new tangent space
        p = p - np.sum(p * x, axis=1, keepdims=True) * x
        
        # Polak-Ribiere update
        g_diff = g_trial - g
        beta = max(0.0, np.sum(g_trial * g_diff) / (np.sum(g * g) + 1e-15))
        
        g = g_trial
        p = -g + beta * p
        E = E_trial
    
    return best_x, best_energy

def _perturb_and_reoptimize(x, sigma=0.1, max_iter=50):
    """Perturb points and re-optimize with short CG."""
    rng = np.random.default_rng(12345)
    # Generate random tangent vectors
    v = rng.normal(size=(N, 3))
    v = v - np.sum(v * x, axis=1, keepdims=True) * x
    v *= sigma
    x_pert = _exp_map_sphere(x, v)
    x_opt, E_opt = _riemannian_cg(x_pert, max_iter=max_iter, tol=1e-8)
    return x_opt, E_opt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Generate diverse initial configurations
    inits = [
        _halton_sphere(N, 2, 3),
        _halton_sphere(N, 3, 5),
        _halton_sphere(N, 5, 7),
        _halton_sphere(N, 2, 5),
        _fibonacci_sphere(N),
        _random_sphere(N, 42),
        _random_sphere(N, 123),
        _random_sphere(N, 999),
    ]
    
    for i in range(len(inits)):
        inits[i] /= np.linalg.norm(inits[i], axis=1, keepdims=True)
    
    # Phase 1: For each init, do tangent-space L-BFGS-B then Riemannian CG polish
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        try:
            # Phase 1a: L-BFGS-B in tangent space (fast initial descent)
            x_tangent, E_tangent = _tangent_space_optimize(x_init, max_iter=200, tol=1e-9)
            
            # Phase 1b: Riemannian CG polish
            x_polished, E_polished = _riemannian_cg(x_tangent, max_iter=100, tol=1e-10)
            
            if E_polished < best_energy:
                best_energy = E_polished
                best_x = x_polished.copy()
                
        except Exception:
            # Fallback: just use Riemannian CG directly
            try:
                x_direct, E_direct = _riemannian_cg(x_init, max_iter=100, tol=1e-10)
                if E_direct < best_energy:
                    best_energy = E_direct
                    best_x = x_direct.copy()
            except Exception:
                E_init, _ = _log_energy_and_grad_sphere(x_init)
                if E_init < best_energy:
                    best_energy = E_init
                    best_x = x_init.copy()
    
    if best_x is None:
        best_x = inits[0]
        best_energy = _log_energy_and_grad_sphere(best_x)[0]
    
    # Phase 2: Basin-hopping-lite: perturb best solution and re-optimize
    if time.time() < deadline - 15:
        for sigma in [0.1, 0.05, 0.02]:
            if time.time() > deadline - 10:
                break
            try:
                x_pert, E_pert = _perturb_and_reoptimize(best_x, sigma=sigma, max_iter=50)
                if E_pert < best_energy:
                    best_energy = E_pert
                    best_x = x_pert.copy()
            except Exception:
                pass
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 5:
        try:
            x_final, E_final = _riemannian_cg(best_x, max_iter=300, tol=1e-12)
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
