"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.special import lpmv, sph_harm

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _log_energy_grad(pts):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-15):
        return -1e18, np.zeros_like(pts)
    energy = -np.sum(np.log(d_upper))
    
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]
    
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _spherical_harmonic_density(pts, l_max=10):
    """
    Estimate density coefficients on sphere using spherical harmonics.
    Returns coefficients c_lm for l in [0, l_max], m in [-l, l].
    """
    n = len(pts)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r[r == 0] = 1e-15
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    
    coeffs = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Y_lm(theta, phi)
            # scipy.special.sph_harm(m, l, phi, theta)
            Y = sph_harm(m, l, phi, theta)
            # Coefficient: <Y_lm, empirical_measure>
            c = np.mean(Y)
            coeffs[(l, m)] = c
    return coeffs

def _density_from_coeffs(coeffs, theta, phi, l_max=10):
    """Compute density from spherical harmonic coefficients."""
    density = np.zeros_like(theta)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if (l, m) in coeffs:
                Y = sph_harm(m, l, phi, theta)
                density += coeffs[(l, m)] * np.real(Y)
    return density

def _optimize_density_coefficients(pts, l_max=10, max_iter=100, lr=0.01):
    """
    Optimize density coefficients to minimize energy-like functional.
    This is a proxy for finding a good distribution.
    """
    # This is a simplified version. In a real implementation, we'd define a functional
    # of the density and optimize its coefficients. Here we just estimate them.
    return _spherical_harmonic_density(pts, l_max)

def _sample_from_density(coeffs, n, l_max=10, seed=42):
    """
    Sample points from a density defined by spherical harmonic coefficients.
    Uses rejection sampling.
    """
    rng = np.random.RandomState(seed)
    points = []
    attempts = 0
    max_attempts = n * 100
    
    # Pre-compute max density for rejection sampling
    # We need to evaluate density on a grid to find max
    theta_grid = np.linspace(0, np.pi, 100)
    phi_grid = np.linspace(-np.pi, np.pi, 100)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    density_grid = _density_from_coeffs(coeffs, theta_mesh, phi_mesh, l_max)
    density_grid = np.maximum(density_grid, 0)
    max_density = np.max(density_grid)
    if max_density < 1e-15:
        max_density = 1.0
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        # Uniform sampling on sphere
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = np.arccos(2 * u - 1)
        phi = 2 * np.pi * v
        
        # Evaluate density at this point
        d = _density_from_coeffs(coeffs, np.array([theta]), np.array([phi]), l_max)
        d_val = d[0]
        
        # Rejection sampling
        if d_val > 0 and rng.uniform(0, max_density) < d_val:
            # Accept point
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append([x, y, z])
    
    if len(points) < n:
        # Fill remaining with random points
        while len(points) < n:
            p = rng.randn(3)
            p /= np.linalg.norm(p)
            points.append(p)
    
    return np.array(points)

def _optimize_discrete(pts0, deadline):
    """Optimize discrete points using L-BFGS-B with sphere projection."""
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e
    
    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()
    
    try:
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
        
        if result.success or result.fun < objective(x0):
            best_pts = result.x.reshape(-1, 3)
            best_pts /= np.linalg.norm(best_pts, axis=1, keepdims=True)
            best_energy = _log_energy(best_pts)
            return best_pts, best_energy
        else:
            return pts0, _log_energy(pts0)
    except Exception:
        return pts0, _log_energy(pts0)

def _optimize_momentum(pts0, deadline, max_iter=3000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.01
    momentum = 0.9
    patience = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
            patience = 0
        else:
            patience += 1
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break
        
        if patience > 50:
            lr = max(lr * 0.5, 1e-10)
            v *= 0.3
            patience = 0
            if lr < 1e-11:
                break
        
        v = momentum * v - lr * grad
        pts_new = pts + v
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts_new = pts_new / norms
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            if e_new < best_energy:
                lr = min(lr * 1.01, 0.05)
        else:
            lr = max(lr * 0.75, 1e-10)
            v *= 0.5
    
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Strategy: Use spherical harmonic density model to generate diverse initializations
    # 1. Start with a few diverse discrete configurations
    # 2. Estimate density from each
    # 3. Sample new points from the density
    # 4. Optimize the sampled points
    
    initial_starts = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123),
        _random_sphere(N, seed=999),
    ]
    
    # For each initial start, estimate density and generate new samples
    for i, pts0 in enumerate(initial_starts):
        if time.time() > deadline - 30:
            break
        
        # Estimate density from this configuration
        coeffs = _spherical_harmonic_density(pts0, l_max=8)
        
        # Sample new points from this density
        new_pts = _sample_from_density(coeffs, N, l_max=8, seed=1000 + i)
        
        # Optimize the new points
        pts_opt, e_opt = _optimize_discrete(new_pts, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
        
        # Refine with momentum
        if time.time() < deadline - 15:
            pts_ref, e_ref = _optimize_momentum(pts_opt, deadline)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref
    
    # Also try direct optimization of original starts as fallback
    for i, pts0 in enumerate(initial_starts):
        if time.time() > deadline - 10:
            break
        
        pts_opt, e_opt = _optimize_discrete(pts0, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Final polish
    if best_pts is not None and time.time() < deadline - 5:
        pts_polished, e_polished = _optimize_momentum(best_pts, deadline, max_iter=1000)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
