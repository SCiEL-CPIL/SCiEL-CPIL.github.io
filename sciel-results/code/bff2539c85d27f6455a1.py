"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = pts.shape[0]
    # Compute pairwise distances vectorized
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0)
    d = np.maximum(d, 1e-15)
    # Use math.fsum for accuracy as specified
    return -math.fsum(math.log(x) for x in d)


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    pts: (N, 3) array
    Returns: (energy, gradient)
    """
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(dist_sq)  # (N, N)
    
    # Energy: sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Vectorized gradient computation
    # grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # dist_sq[i,j] = |x_i - x_j|^2
    # We want grad[i] = sum_j (pts[i] - pts[j]) / dist_sq[i,j] for j != i
    
    # Create inverse distance squared matrix, zero out diagonal
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    # Avoid division by zero
    safe_dist_sq = np.where(dist_sq > 1e-24, dist_sq, 1e24)
    inv_dist_sq[mask] = 1.0 / safe_dist_sq[mask]
    
    # grad[i] = sum_j (pts[i] - pts[j]) * inv_dist_sq[i,j]
    #         = pts[i] * sum_j inv_dist_sq[i,j] - sum_j pts[j] * inv_dist_sq[i,j]
    #         = pts[i] * row_sums - pts @ inv_dist_sq.T
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    grad = pts * row_sums[:, np.newaxis] - pts @ inv_dist_sq.T  # (N, 3)
    
    # Project gradient onto tangent space: grad_tan = grad - (grad . x_i) * x_i
    dots = np.sum(grad * pts, axis=1)  # (N,)
    grad = grad - dots[:, np.newaxis] * pts
    
    return energy, grad


def _spherical_energy_and_grad(params):
    """Unpack spherical coordinates, compute energy and gradient.
    params: (2N,) array of (theta_i, phi_i) for each point
    """
    thetas = params[0::2]
    phis = params[1::2]
    
    # Convert to Cartesian
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    pts = np.column_stack([x, y, z])
    
    # Compute energy
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-15)
    energy = -np.sum(np.log(d))
    
    # Vectorized Cartesian gradient
    n = N
    mask = ~np.eye(n, dtype=bool)
    safe_dist_sq = np.where(dist_sq > 1e-24, dist_sq, 1e24)
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / safe_dist_sq[mask]
    
    row_sums = np.sum(inv_dist_sq, axis=1)
    grad_cart = pts * row_sums[:, np.newaxis] - pts @ inv_dist_sq.T
    
    # Convert gradient to spherical coordinates
    # dE/dtheta = grad_cart . dpts/dtheta
    # dE/dphi = grad_cart . dpts/dphi
    grad_params = np.zeros_like(params)
    
    # Vectorized computation of dtheta and dphi components
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)
    cos_phis = np.cos(phis)
    sin_phis = np.sin(phis)
    
    # dpts/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
    # dpts/dphi = (-sin(theta)sin(phi), sin(theta)cos(phi), 0)
    
    # dE/dtheta_i = grad_cart[i] . dpts_i/dtheta
    # = grad_cart[i,0]*cos(th_i)*cos(ph_i) + grad_cart[i,1]*cos(th_i)*sin(ph_i) + grad_cart[i,2]*(-sin(th_i))
    grad_params[0::2] = (grad_cart[:, 0] * cos_thetas * cos_phis + 
                         grad_cart[:, 1] * cos_thetas * sin_phis - 
                         grad_cart[:, 2] * sin_thetas)
    
    # dE/dphi_i = grad_cart[i] . dpts_i/dphi
    # = grad_cart[i,0]*(-sin(th_i)*sin(ph_i)) + grad_cart[i,1]*sin(th_i)*cos(ph_i) + 0
    grad_params[1::2] = (-grad_cart[:, 0] * sin_thetas * sin_phis + 
                         grad_cart[:, 1] * sin_thetas * cos_phis)
    
    return energy, grad_params


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _cartesian_to_spherical(pts):
    """Convert Cartesian points to spherical coordinates (theta, phi)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    thetas = np.arccos(np.clip(z, -1.0, 1.0))
    phis = np.arctan2(y, x)
    params = np.zeros(2 * N)
    params[0::2] = thetas
    params[1::2] = phis
    return params


def _spherical_to_cartesian(params):
    """Convert spherical coordinates (theta, phi) to Cartesian points."""
    thetas = params[0::2]
    phis = params[1::2]
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])


def _optimize_single_start(start_pts, time_limit):
    """Run L-BFGS-B optimization from a given start point."""
    start_time = time.time()
    params0 = _cartesian_to_spherical(start_pts)
    
    # Define bounds for theta and phi
    # theta in [0, pi], phi in [0, 2*pi)
    bounds = [(0, np.pi)] * N + [(0, 2 * np.pi)] * N
    # Interleave bounds correctly: (theta_1, phi_1, theta_2, phi_2, ...)
    bounds = []
    for i in range(N):
        bounds.append((0, np.pi))  # theta
        bounds.append((0, 2 * np.pi))  # phi
        
    result = minimize(_spherical_energy_and_grad, params0, method='L-BFGS-B',
                      jac=True, bounds=bounds,
                      options={'maxiter': 5000, 'maxfun': 100000,
                               'ftol': 1e-12, 'gtol': 1e-10,
                               'disp': False})
    
    if result.success or result.fun < _log_energy(start_pts):
        pts_opt = _spherical_to_cartesian(result.x)
        pts_opt = _project_to_sphere(pts_opt)
        return pts_opt
    
    return start_pts


def construct():
    # Time limit: 150s, keep safety margin of 10s
    start_time = time.time()
    time_limit = 140.0  # safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # Generate multiple initial configurations
    # 1. Fibonacci spiral
    pts_fib = _fibonacci_sphere(N)
    # 2. Random points (seed 1)
    pts_rand1 = _random_sphere(N, seed=1)
    # 3. Random points (seed 2)
    pts_rand2 = _random_sphere(N, seed=2)
    # 4. Fibonacci with noise
    pts_fib_noise = pts_fib + 0.05 * np.random.default_rng(3).normal(size=(N, 3))
    pts_fib_noise = _project_to_sphere(pts_fib_noise)
    
    initial_configs = [pts_fib, pts_rand1, pts_rand2, pts_fib_noise]
    
    for i, start_pts in enumerate(initial_configs):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10:  # Leave 10s for final polish and checks
            break
            
        # Run optimization
        try:
            optimized_pts = _optimize_single_start(start_pts, time_limit)
            energy = _log_energy(optimized_pts)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = optimized_pts
        except Exception:
            pass
    
    # If no optimization succeeded or best_pts is None, use the best initial config
    if best_pts is None:
        # Evaluate initial configs
        for start_pts in initial_configs:
            e = _log_energy(start_pts)
            if e < best_energy:
                best_energy = e
                best_pts = start_pts
    
    # Final polish: Use L-BFGS-B again with tighter tolerance on the best point found
    elapsed = time.time() - start_time
    if elapsed < time_limit - 5:
        try:
            params_best = _cartesian_to_spherical(best_pts)
            bounds = []
            for i in range(N):
                bounds.append((0, np.pi))
                bounds.append((0, 2 * np.pi))
                
            # Tighter tolerance for polish
            result_polish = minimize(_spherical_energy_and_grad, params_best, method='L-BFGS-B',
                                     jac=True, bounds=bounds,
                                     options={'maxiter': 2000, 'maxfun': 50000,
                                              'ftol': 1e-14, 'gtol': 1e-12,
                                              'disp': False})
            
            if result_polish.success:
                pts_polish = _spherical_to_cartesian(result_polish.x)
                pts_polish = _project_to_sphere(pts_polish)
                e_polish = _log_energy(pts_polish)
                if e_polish < best_energy:
                    best_pts = pts_polish
                    best_energy = e_polish
        except Exception:
            pass
            
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points and nudge apart if needed
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge apart
                perp = best_pts[i] - np.dot(best_pts[i], best_pts[j]) * best_pts[j]
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-15:
                    best_pts[j] = best_pts[j] + 1e-8 * perp / perp_norm
                else:
                    # Use a different direction
                    best_pts[j] = best_pts[j] + 1e-8 * np.array([1.0, 0.0, 0.0])
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    # Final check
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
