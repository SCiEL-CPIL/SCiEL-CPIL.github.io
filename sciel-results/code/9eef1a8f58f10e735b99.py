"""Rewritten program: L-BFGS on spherical coordinates + direct gradient polishing."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _energy_fast(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|) using vectorized ops."""
    sq_dists = 2.0 - 2.0 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-15)
    iu = np.triu_indices(N, k=1)
    log_dists_sq = np.log(sq_dists[iu])
    return -0.5 * math.fsum(log_dists_sq.tolist())

def _gradient(pts):
    """Compute gradient of logarithmic energy, projected onto tangent space."""
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    weights = 1.0 / dists_sq
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.sum(diffs * weights[..., np.newaxis], axis=1)
    dot_products = np.sum(grad_raw * pts, axis=1)
    grad = grad_raw - dot_products[:, np.newaxis] * pts
    return grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _spherical_to_cartesian(theta, phi):
    """Convert spherical coords (theta in [0,pi], phi in [0,2pi]) to Cartesian."""
    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

def _cartesian_to_spherical(pts):
    """Convert Cartesian points on unit sphere to (theta, phi)."""
    z = pts[:, 2]
    x = pts[:, 0]
    y = pts[:, 1]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return theta, phi

def _energy_and_grad_spherical(theta, phi):
    """Energy and gradient in spherical coordinates."""
    pts = _spherical_to_cartesian(theta, phi)
    # Compute energy
    sq_dists = 2.0 - 2.0 * (pts @ pts.T)
    np.fill_diagonal(sq_dists, 1.0)
    sq_dists = np.maximum(sq_dists, 1e-15)
    iu = np.triu_indices(N, k=1)
    log_dists_sq = np.log(sq_dists[iu])
    energy = -0.5 * math.fsum(log_dists_sq.tolist())
    
    # Compute gradient in Cartesian
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diffs**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    weights = 1.0 / dists_sq
    np.fill_diagonal(weights, 0.0)
    grad_cart = np.sum(diffs * weights[..., np.newaxis], axis=1)
    
    # Project to tangent space
    dot_products = np.sum(grad_cart * pts, axis=1)
    grad_tangent = grad_cart - dot_products[:, np.newaxis] * pts
    
    # Convert tangent gradient to spherical gradient
    # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
    # dx/dtheta = cos(theta)cos(phi), dy/dtheta = cos(theta)sin(phi), dz/dtheta = -sin(theta)
    # dx/dphi = -sin(theta)sin(phi), dy/dphi = sin(theta)cos(phi), dz/dphi = 0
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    
    grad_theta = grad_tangent[:, 0] * ct * cp + grad_tangent[:, 1] * ct * sp + grad_tangent[:, 2] * (-st)
    grad_phi = grad_tangent[:, 0] * (-st * sp) + grad_tangent[:, 1] * (st * cp)
    
    # Combine into flat vector
    grad = np.concatenate([grad_theta, grad_phi])
    return energy, grad

def _lbfgs_spherical(theta, phi, time_limit, start_time):
    """Run L-BFGS-B on spherical coordinates."""
    x0 = np.concatenate([theta, phi])
    
    def fun(x):
        if time.time() - start_time > time_limit:
            return float('inf'), np.zeros_like(x)
        t = x[:N]
        p = x[N:]
        return _energy_and_grad_spherical(t, p)
    
    result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': 10000, 'maxfun': 10000, 'maxls': 50,
                               'gtol': 1e-10, 'ftol': 1e-10, 'iprint': 0})
    
    if result.success or result.fun < float('inf'):
        t_opt = result.x[:N]
        p_opt = result.x[N:]
        pts = _spherical_to_cartesian(t_opt, p_opt)
        return pts, result.fun
    return None, float('inf')

def _adam_optimize(pts, max_iter=5000, time_limit=None, start_time=None, lr_init=0.05):
    """Adam optimizer for sphere-constrained optimization."""
    if start_time is None:
        start_time = time.time()
    
    pts = _project_to_sphere(pts.copy())
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    
    decay = 0.9995
    
    for i in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        grad = _gradient(pts)
        
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        pts_new = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
        pts_new = _project_to_sphere(pts_new)
        
        energy = _energy_fast(pts_new)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
        
        pts = pts_new
        lr *= decay
        
    return best_pts, best_energy

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _sobol_sphere(n):
    """Use scipy's Sobol sequence for quasi-random initialization."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=123)
    samples = sampler.random(n)
    # Map from [0,1]^3 to sphere
    # Use spherical coordinates
    theta = np.pi * samples[:, 0]
    phi = 2.0 * np.pi * samples[:, 1]
    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

def _refine_direct(pts, time_limit, start_time):
    """Refine using direct gradient descent with adaptive step size."""
    pts = _project_to_sphere(pts.copy())
    best_energy = _energy_fast(pts)
    best_pts = pts.copy()
    
    step_size = 0.1
    patience = 0
    max_patience = 200
    
    while time.time() - start_time < time_limit:
        grad = _gradient(pts)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-12:
            break
        
        # Adaptive step size
        pts_new = pts - step_size * grad
        pts_new = _project_to_sphere(pts_new)
        
        energy = _energy_fast(pts_new)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts_new.copy()
            pts = pts_new
            patience = 0
            # Increase step size slightly if improving
            step_size = min(step_size * 1.05, 0.5)
        else:
            patience += 1
            step_size *= 0.9
            if patience > max_patience:
                break
    
    return best_pts, best_energy

def construct():
    deadline = time.time() + 145  # Leave 5s margin
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: L-BFGS on spherical coordinates from Fibonacci
    pts_fib = _fibonacci_sphere(N)
    theta, phi = _cartesian_to_spherical(pts_fib)
    t0 = time.time()
    remaining = deadline - t0 - 10
    if remaining > 5:
        opts_pts, opts_energy = _lbfgs_spherical(theta, phi, time_limit=min(remaining, 60), start_time=t0)
        if opts_pts is not None and opts_energy < best_energy:
            best_energy = opts_energy
            best_pts = opts_pts
    
    # Phase 2: L-BFGS from Sobol initialization
    if time.time() < deadline - 15:
        pts_sobol = _sobol_sphere(N)
        theta, phi = _cartesian_to_spherical(pts_sobol)
        t0 = time.time()
        remaining = deadline - t0 - 10
        if remaining > 5:
            opts_pts, opts_energy = _lbfgs_spherical(theta, phi, time_limit=min(remaining, 50), start_time=t0)
            if opts_pts is not None and opts_energy < best_energy:
                best_energy = opts_energy
                best_pts = opts_pts
    
    # Phase 3: L-BFGS from random initialization
    if time.time() < deadline - 15:
        pts_rand = _random_sphere(N, seed=12345)
        theta, phi = _cartesian_to_spherical(pts_rand)
        t0 = time.time()
        remaining = deadline - t0 - 10
        if remaining > 5:
            opts_pts, opts_energy = _lbfgs_spherical(theta, phi, time_limit=min(remaining, 50), start_time=t0)
            if opts_pts is not None and opts_energy < best_energy:
                best_energy = opts_energy
                best_pts = opts_pts
    
    # Phase 4: Adam refinement on best solution
    if best_pts is not None and time.time() < deadline - 10:
        t0 = time.time()
        remaining = deadline - t0 - 5
        if remaining > 5:
            opts_pts, opts_energy = _adam_optimize(best_pts, max_iter=3000, time_limit=remaining, start_time=t0, lr_init=0.02)
            if opts_energy < best_energy:
                best_energy = opts_energy
                best_pts = opts_pts
    
    # Phase 5: Direct gradient refinement
    if best_pts is not None and time.time() < deadline - 8:
        t0 = time.time()
        remaining = deadline - t0 - 3
        if remaining > 3:
            opts_pts, opts_energy = _refine_direct(best_pts, time_limit=remaining, start_time=t0)
            if opts_energy < best_energy:
                best_energy = opts_energy
                best_pts = opts_pts
    
    # Final polish with very small Adam steps
    if best_pts is not None and time.time() < deadline - 5:
        t0 = time.time()
        remaining = deadline - t0 - 2
        if remaining > 2:
            opts_pts, opts_energy = _adam_optimize(best_pts, max_iter=1000, time_limit=remaining, start_time=t0, lr_init=0.005)
            if opts_energy < best_energy:
                best_energy = opts_energy
                best_pts = opts_pts
    
    return _project_to_sphere(best_pts)

# EVOLVE-BLOCK-END
