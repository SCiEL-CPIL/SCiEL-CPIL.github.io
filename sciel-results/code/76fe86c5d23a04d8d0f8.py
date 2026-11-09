"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _energy_and_grad_spherical(theta_phi):
    """
    theta_phi: (n, 2) array of (theta, phi) angles
    Returns (energy, gradient in (theta, phi) space)
    Combined computation for efficiency.
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    n = len(theta)
    
    # Convert to Cartesian
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    
    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    pts = np.column_stack([x, y, z])
    
    # Pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    dists = np.maximum(dists, 1e-15)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = float(math.fsum(-np.log(dists[iu])))
    
    # Gradient in Cartesian: grad[i] = sum_j (x_i - x_j) / d_ij^2
    inv_d2 = 1.0 / (dists ** 2)
    inv_d2[np.diag_indices(n)] = 0.0
    grad_cart = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    # Chain rule to spherical coordinates
    # dtheta/dx components: dx/dtheta = cos(theta)cos(phi), dy/dtheta = cos(theta)sin(phi), dz/dtheta = -sin(theta)
    # dphi/dx components: dx/dphi = -sin(theta)sin(phi), dy/dphi = sin(theta)cos(phi), dz/dphi = 0
    
    grad_theta = grad_cart[:, 0] * cos_t * cos_p + grad_cart[:, 1] * cos_t * sin_p + grad_cart[:, 2] * (-sin_t)
    grad_phi = grad_cart[:, 0] * (-sin_t * sin_p) + grad_cart[:, 1] * (sin_t * cos_p)
    
    grad = np.column_stack([grad_theta, grad_phi])
    return energy, grad

def _fibonacci_spiral():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts

def _pts_to_spherical(pts):
    """Convert Cartesian points to (theta, phi) spherical coordinates."""
    theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
    phi_s = np.arctan2(pts[:, 1], pts[:, 0])
    return np.column_stack([theta, phi_s])

def _spherical_to_pts(tp):
    """Convert (theta, phi) to Cartesian points on unit sphere."""
    theta = tp[:, 0]
    phi = tp[:, 1]
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    pts = np.column_stack([x, y, z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _random_sphere_points(seed):
    """Generate random points on sphere with given seed."""
    rng = np.random.default_rng(seed)
    # Uniform on sphere using normal distribution
    v = rng.normal(size=(N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    pts = v / norms
    return pts

def _ring_initialization():
    """Generate ring-based initialization for diversity."""
    pts_list = []
    # Multiple rings at different z-levels
    n_rings = 10
    pts_per_ring = N // n_rings
    remainder = N - pts_per_ring * n_rings
    
    for ring_idx in range(n_rings):
        z = 1.0 - 2.0 * (ring_idx + 0.5) / n_rings
        r = np.sqrt(max(0.0, 1.0 - z * z))
        n_in_ring = pts_per_ring + (1 if ring_idx < remainder else 0)
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False) + ring_idx * 0.3
        for a in angles:
            pts_list.append([r * np.cos(a), r * np.sin(a), z])
    
    pts = np.array(pts_list[:N])
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    from scipy.optimize import minimize
    
    best_energy = np.inf
    best_tp = None
    
    # Objective and gradient functions
    def objective(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _energy_and_grad_spherical(tp_2d)[0]
    
    def gradient(tp_flat):
        tp_2d = tp_flat.reshape(-1, 2)
        return _energy_and_grad_spherical(tp_2d)[1].ravel()
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    pts_fib = _fibonacci_spiral()
    tp_fib = _pts_to_spherical(pts_fib)
    initializations.append(('fib', tp_fib))
    
    # 2. Ring-based
    pts_ring = _ring_initialization()
    tp_ring = _pts_to_spherical(pts_ring)
    initializations.append(('ring', tp_ring))
    
    # 3. Random with different seeds
    for seed in [42, 123, 456]:
        pts_rand = _random_sphere_points(seed)
        tp_rand = _pts_to_spherical(pts_rand)
        initializations.append((f'rand_{seed}', tp_rand))
    
    # Optimize each initialization
    for name, tp_init in initializations:
        if time.time() > deadline - 10.0:
            break
        
        e_init, _ = _energy_and_grad_spherical(tp_init)
        
        # L-BFGS-B optimization
        result = minimize(
            objective,
            tp_init.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_tp = result.x.reshape(-1, 2)
    
    # Final high-precision polish on the best result
    if best_tp is not None and time.time() < deadline - 5.0:
        result_polish = minimize(
            objective,
            best_tp.ravel(),
            jac=gradient,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'maxfun': 20000,
                'ftol': 1e-15,
                'gtol': 1e-13,
                'maxls': 100
            }
        )
        
        if result_polish.fun < best_energy:
            best_energy = result_polish.fun
            best_tp = result_polish.x.reshape(-1, 2)
    
    # Convert back to Cartesian
    pts = _spherical_to_pts(best_tp)
    
    return pts
# EVOLVE-BLOCK-END
