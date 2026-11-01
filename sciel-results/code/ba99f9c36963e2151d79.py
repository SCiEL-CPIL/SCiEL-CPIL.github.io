"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log(|xi-xj|)"""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    dist = np.sqrt(d2)
    iu = np.triu_indices(n, k=1)
    log_d = np.log(dist[iu])
    return -math.fsum(log_d)

def _gradient_log_energy(pts):
    """Tangent-space gradient of logarithmic energy, fully vectorized."""
    n = pts.shape[0]
    dots = pts @ pts.T
    d2 = 2.0 - 2.0 * dots
    np.fill_diagonal(d2, 1.0)
    d2 = np.maximum(d2, 1e-15)
    
    inv_d2 = 1.0 / d2
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_inv_d2 = inv_d2.sum(axis=1)
    sum_xj_inv_d2 = (pts.T @ inv_d2).T
    
    # grad of E = -sum_{i<j} log(d_ij): dE/dx_i = -sum_{j!=i} (x_i - x_j)/d_ij^2
    grad = -(pts * sum_inv_d2[:, np.newaxis] - sum_xj_inv_d2)
    
    # Project to tangent space
    grad_dot_xi = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tan = grad - grad_dot_xi * pts
    
    return grad_tan

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / np.maximum(norms, 1e-15)

def _perturb_sphere(pts, rng, sigma=0.05):
    """Perturb points and renormalize."""
    noise = rng.standard_normal(pts.shape) * sigma
    new_pts = pts + noise
    return _project_to_sphere(new_pts)

def _riemannian_descent(pts, max_iter=800, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    n = pts.shape[0]
    pts = _project_to_sphere(pts.copy())
    f = _log_energy(pts)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_log_energy(pts)
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Initial step size
        step = 1.0 / (1.0 + 0.5 * grad_norm)
        
        accepted = False
        for ls_iter in range(40):
            new_pts = pts - step * grad
            new_pts = _project_to_sphere(new_pts)
            new_f = _log_energy(new_pts)
            
            if np.isfinite(new_f) and new_f < f:
                if new_f < f - 1e-6 * step * grad_norm:
                    accepted = True
                    break
                if new_f < f - 1e-10:
                    accepted = True
                    break
            step *= 0.7
        
        if accepted:
            pts = new_pts
            f = new_f
        else:
            break
    
    return pts, f

def _lbfgs_optimize(pts0, deadline, maxiter=500, gtol=1e-8):
    """Run L-BFGS-B optimization from a given starting configuration."""
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _log_energy(pts)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_log_energy(pts)
        return g.reshape(-1)
    
    x0 = pts0.reshape(-1)
    
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxfun': 2000, 'ftol': 1e-10}
        )
        if res.fun < float('inf'):
            pts_new = _project_to_sphere(res.x.reshape(N, 3))
            e_new = _log_energy(pts_new)
            if np.isfinite(e_new):
                return pts_new, e_new
    except Exception:
        pass
    
    return pts0, _log_energy(pts0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_points = None
    best_energy = float('inf')
    
    # Generate diverse starting points
    starts = []
    
    # Start 1: Pure Fibonacci
    fib_pts = _fibonacci_sphere(N)
    starts.append(fib_pts)
    
    # Starts 2-14: Fibonacci with random perturbations at various scales
    rng = np.random.default_rng(123)
    scales = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    for scale_idx, scale in enumerate(scales):
        for seed in [42 + scale_idx * 100, 43 + scale_idx * 100]:
            if time.time() > deadline - 20.0:
                break
            perturbed = _perturb_sphere(fib_pts, rng, sigma=scale)
            starts.append(perturbed)
    
    # Starts 15-19: Pure random starts
    for seed in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 20.0:
            break
        random_pts = _random_sphere(N, rng)
        starts.append(random_pts)
    
    # Run optimization from each start, tracking best
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        # L-BFGS-B pass
        pts, energy = _lbfgs_optimize(start_pts, deadline - 25.0, maxiter=400, gtol=1e-8)
        
        if energy < best_energy:
            best_energy = energy
            best_points = pts.copy()
    
    # Final polish: Riemannian gradient descent from best point
    if best_points is not None and time.time() < deadline - 15.0:
        pts_rd, e_rd = _riemannian_descent(best_points, max_iter=500, tol=1e-12, deadline=deadline - 15.0)
        if e_rd < best_energy:
            best_energy = e_rd
            best_points = pts_rd
    
    # Final L-BFGS-B polish with tighter tolerance
    if best_points is not None and time.time() < deadline - 10.0:
        pts_final, e_final = _lbfgs_optimize(best_points, deadline - 5.0, maxiter=800, gtol=1e-12)
        if e_final < best_energy:
            best_energy = e_final
            best_points = pts_final
    
    # Ensure valid output
    if best_points is None:
        best_points = fib_pts
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        best_points = fib_pts
    
    # Normalize
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
