"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on unit sphere.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so (x_i - x_j)/|x_i - x_j|^2 = diff[i,j] / dist[i,j]^2
    # grad[i] = sum_j diff[i,j] / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _spherical_minimize(x0, time_limit, maxiter=5000):
    """Run L-BFGS-B on flattened coordinates, projecting to sphere each iteration."""
    start_time = time.time()
    best_x = x0.copy()
    best_energy = _energy_and_grad(x0)[0]
    
    def fun(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space: g - (g·x)x
        g_proj = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g_proj.flatten()
    
    x_flat = x0.flatten()
    
    result = minimize(fun, x_flat, jac=jac, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'maxfun': 20000,
                               'ftol': 1e-12, 'gtol': 1e-10, 'iprint': 0})
    
    # Project final result
    pts = _project_to_sphere(result.x.reshape(N, 3))
    final_energy = _energy_and_grad(pts)[0]
    
    if final_energy < best_energy:
        best_x = pts
        best_energy = final_energy
    
    return best_x, best_energy, time.time() - start_time


def _polish(pts, time_limit, maxiter=2000):
    """High-precision polish using Newton-like method in tangent space."""
    start_time = time.time()
    x = pts.copy()
    n = len(x)
    
    for it in range(maxiter):
        if time.time() - start_time > time_limit:
            break
        
        e, g = _energy_and_grad(x)
        
        # Project gradient to tangent space
        g_tan = g - np.sum(g * x, axis=1, keepdims=True) * x
        
        # Compute Hessian approximation in tangent space (crude)
        # Use a simple gradient step with adaptive step size
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        # Adaptive step size
        step = 0.1 / (1.0 + g_norm * 0.1)
        x_new = x - step * g_tan
        x_new = _project_to_sphere(x_new)
        
        e_new, _ = _energy_and_grad(x_new)
        if e_new < e:
            x = x_new
        else:
            step *= 0.5
            x_new = _project_to_sphere(x - step * g_tan)
            e_new, _ = _energy_and_grad(x_new)
            if e_new < e:
                x = x_new
            else:
                break
    
    return x, _energy_and_grad(x)[0]


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed):
    """Generate random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    deadline = time.time() + 145.0  # 5 second safety margin
    
    # Generate multiple initial configurations
    configs = []
    configs.append(_fibonacci_spiral(N))
    
    # Add random restarts
    for seed in range(5):
        configs.append(_random_points(N, seed))
    
    # Also try perturbed fibonacci
    rng = np.random.RandomState(42)
    for _ in range(3):
        pert = rng.randn(N, 3) * 0.1
        cfg = _project_to_sphere(_fibonacci_spiral(N) + pert)
        configs.append(cfg)
    
    best_pts = configs[0]
    best_energy = _energy_and_grad(best_pts)[0]
    
    # Optimize each configuration
    for idx, cfg in enumerate(configs):
        if time.time() > deadline - 10.0:
            break
        
        time_left = deadline - time.time() - 2.0
        if time_left < 5.0:
            break
        
        # Use L-BFGS with time limit
        try:
            pts, energy, elapsed = _spherical_minimize(cfg, time_limit=time_left * 0.6)
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            continue
        
        # Polish if time remains
        if time.time() < deadline - 5.0:
            try:
                pts, energy = _polish(pts, time_limit=min(5.0, deadline - time.time() - 2.0))
                if energy < best_energy:
                    best_energy = energy
                    best_pts = pts.copy()
            except Exception:
                pass
    
    # Final verification
    final_pts = _project_to_sphere(best_pts)
    final_energy, _ = _energy_and_grad(final_pts)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(final_pts[i] - final_pts[j])
            if d < 1e-9:
                # Separate them slightly
                final_pts[j] = final_pts[i] + 1e-8 * np.array([1, 0, 0])
                final_pts[j] /= np.linalg.norm(final_pts[j])
    
    # Verify finite values
    if not np.all(np.isfinite(final_pts)):
        return _fibonacci_spiral(N)
    
    return final_pts
# EVOLVE-BLOCK-END
