"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    Gradient w.r.t. x_k = 2 * sum_{j!=k} x_j / (2 - 2*x_k.x_j)
    """
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)
    
    iu = np.triu_indices(N, 1)
    d2_upper = dist2[iu]
    energy = -0.5 * np.sum(np.log(d2_upper))
    
    inv_d2 = np.zeros_like(dots)
    mask = ~np.eye(N, dtype=bool)
    inv_d2[mask] = 1.0 / dist2[mask]
    
    grad = 2.0 * (inv_d2 @ pts)
    return energy, grad


def _energy_only(pts):
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(N, 1)
    d2_upper = dist2[iu]
    return -0.5 * np.sum(np.log(d2_upper))


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _tangent_grad_descent(pts, max_iter=500, tol=1e-10, deadline=None):
    """Gradient descent on the sphere using tangent-space projection."""
    current = pts.copy()
    current = _normalize(current)
    best_energy = _energy_only(current)
    best_pts = current.copy()
    
    step_size = 0.01
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(current)
        
        # Project gradient to tangent space: g_tan = grad - (grad . x) * x
        grad_proj = grad - np.sum(grad * current, axis=1, keepdims=True) * current
        
        # Check convergence
        max_grad = np.max(np.linalg.norm(grad_proj, axis=1))
        if max_grad < tol:
            break
        
        # Take step and re-normalize
        current = current - step_size * grad_proj
        current = _normalize(current)
        
        # Adaptive step size
        new_energy = _energy_only(current)
        if new_energy < energy:
            step_size = min(step_size * 1.2, 0.1)
            if new_energy < best_energy:
                best_energy = new_energy
                best_pts = current.copy()
        else:
            step_size *= 0.5
            current = best_pts  # revert
    
    return best_pts, best_energy


def _fibonacci_init():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_init(rng):
    pts = rng.normal(0, 1, size=(N, 3))
    return _normalize(pts)


def _sobol_init():
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=2, scramble=True, seed=42)
    samples = sampler.random(N)
    # Map 2D [0,1)^2 to sphere: theta = arccos(2*y-1), phi = 2*pi*x
    theta = np.arccos(2.0 * samples[:, 1] - 1.0)
    phi = 2.0 * np.pi * samples[:, 0]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_pts = _fibonacci_init()
    
    # Generate diverse initial points
    inits = [
        _fibonacci_init(),
        _sobol_init(),
        _random_init(rng),
        _random_init(rng),
    ]
    
    # Add perturbed versions of fibonacci
    for _ in range(3):
        pert = _fibonacci_init() + rng.normal(0, 0.3, size=(N, 3))
        inits.append(_normalize(pert))
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
        
        # Phase 1: L-BFGS-B with sphere constraint handled by re-normalization
        try:
            def obj_and_grad(x):
                pts = x.reshape(N, 3)
                pts = _normalize(pts)
                e, g = _energy_and_grad(pts)
                return e, g.ravel()
            
            result = minimize(
                obj_and_grad[0],
                init_pts.ravel(),
                jac=obj_and_grad[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_pts = _normalize(result.x.reshape(N, 3))
            
            # Check validity
            if not np.all(np.isfinite(opt_pts)):
                continue
            
            dots = opt_pts @ opt_pts.T
            dist2 = 2.0 - 2.0 * dots
            np.fill_diagonal(dist2, 1.0)
            min_dist2 = np.min(dist2)
            if min_dist2 < 1e-18:
                continue
            
            opt_energy = _energy_only(opt_pts)
            
            # Phase 2: Tangent-space gradient descent polish
            polished, polished_energy = _tangent_grad_descent(
                opt_pts, max_iter=300, tol=1e-12, 
                deadline=deadline - 5.0
            )
            
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pts = polished.copy()
                
        except Exception:
            continue
    
    # Final high-precision polish on best result
    if time.time() < deadline - 3.0:
        final_pts, final_energy = _tangent_grad_descent(
            best_pts, max_iter=500, tol=1e-14,
            deadline=deadline - 1.0
        )
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    best_pts = _normalize(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
