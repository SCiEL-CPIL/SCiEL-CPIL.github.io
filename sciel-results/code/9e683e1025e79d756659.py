"""Optimized 100-point logarithmic energy minimizer on the unit sphere."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import math
import time

N = 100
DEADLINE = 150.0  # seconds

def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _log_energy_and_grad(pts):
    """Compute logarithmic energy and its gradient.
    
    Energy = sum_{i<j} log(1 / ||x_i - x_j||) = -sum_{i<j} log(||x_i - x_j||)
    
    Returns (energy, gradient) where gradient is shape (n, 3).
    """
    n = pts.shape[0]
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # We compute full sum then halve (since i<j and j<i are symmetric)
    log_dist = np.log(dist)
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    energy = -np.sum(log_dist[mask]) / 2.0
    
    # Gradient computation
    # d/dx_i [ -log(||x_i - x_j||) ] = -(x_i - x_j) / ||x_i - x_j||^2
    # For point i: sum over j != i of -(x_i - x_j) / dist_ij^2
    #              = sum over j != i of (x_j - x_i) / dist_ij^2
    
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j (x_j - x_i) / dist_ij^2 = -sum_j (x_i - x_j) / dist_ij^2
    #        = -sum_j diff[i,j] / dist[i,j]^2
    
    # dist^2
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)  # avoid div by zero
    
    # grad[i] = -sum_j diff[i,j] / dist_sq[i,j]
    # diff[i,j] / dist_sq[i,j] has shape (n, n, 3)
    # We want sum over j for each i
    
    # Normalize: diff / dist_sq
    norm_diff = diff / dist_sq[:, :, np.newaxis]  # (n, n, 3)
    # Set diagonal to zero
    norm_diff[np.arange(n), np.arange(n), :] = 0.0
    # grad[i] = -sum_j norm_diff[i,j]
    grad = -np.sum(norm_diff, axis=1)  # (n, 3)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms

def _optimize_from_start(start_pts, time_budget):
    """Run L-BFGS-B optimization from a starting configuration."""
    start_time = time.time()
    
    # Flatten points for optimizer
    x0 = start_pts.flatten()
    
    def objective_and_grad(x):
        pts = x.reshape(-1, 3)
        # Project to sphere to maintain constraint
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        # Chain rule for projection: gradient in flattened space
        # For projection p/norm(p), the gradient w.r.t. x is:
        # g_proj = (I - outer(p_hat, p_hat)) @ g
        # where p_hat = p / ||p||
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        p_hat = pts / norms
        # Project gradient onto tangent space
        g_flat = g.flatten()
        # For each point, project gradient
        proj_grad = np.zeros_like(g)
        for i in range(N):
            proj_grad[i] = g[i] - np.dot(g[i], p_hat[i]) * p_hat[i]
        return e, proj_grad.flatten()
    
    # Use L-BFGS-B with projection
    # We'll use a simpler approach: optimize in Cartesian space with projection at each step
    def fun(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(-1, 3))
        e, g = _log_energy_and_grad(pts)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        p_hat = pts / norms
        proj_grad = np.zeros_like(g)
        for i in range(N):
            proj_grad[i] = g[i] - np.dot(g[i], p_hat[i]) * p_hat[i]
        return proj_grad.flatten()
    
    result = minimize(
        fun, x0, jac=jac, method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    
    elapsed = time.time() - start_time
    best_pts = _project_to_sphere(result.x.reshape(-1, 3))
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    return best_pts, best_energy, elapsed

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5 second safety margin
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Perturbed Fibonacci spirals
    rng = np.random.default_rng(42)
    for trial in range(10):
        pts = _fibonacci_spiral(N)
        noise = rng.normal(0, 0.1, size=pts.shape)
        starts.append(_project_to_sphere(pts + noise))
    
    # 3. Random starts
    for trial in range(5):
        pts = rng.normal(0, 1, size=(N, 3))
        starts.append(_project_to_sphere(pts))
    
    # 4. Spherical harmonic based starts
    # Create points using different frequencies
    for freq in [3, 5, 7]:
        theta = np.linspace(0, np.pi, N)
        phi = np.linspace(0, 2*np.pi, N, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        # Add some variation
        TH = TH + 0.1 * np.sin(5 * PH)
        pts = np.column_stack([
            np.sin(TH) * np.cos(PH),
            np.sin(TH) * np.sin(PH),
            np.cos(TH)
        ]).flatten().reshape(-1, 3)
        if len(pts) >= N:
            pts = pts[:N]
            starts.append(_project_to_sphere(pts))
    
    best_pts = None
    best_energy = float('inf')
    
    # Optimize from each start
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Estimate time budget for this start
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        
        # Allocate time based on remaining starts
        num_remaining = len(starts) - idx
        time_per_start = min(remaining / num_remaining, 15.0)
        
        try:
            pts, energy, elapsed = _optimize_from_start(start_pts, time_per_start)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Final polish on the best result
    if best_pts is not None and time.time() < deadline - 2.0:
        try:
            polished_pts, polished_energy, _ = _optimize_from_start(best_pts, deadline - time.time() - 1.0)
            if polished_energy < best_energy:
                best_pts = polished_pts
                best_energy = polished_energy
        except Exception:
            pass
    
    # Ensure validity: check for near-coincident points
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final projection and validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for points too close together
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] = best_pts[i] + 1e-8 * best_pts[i]
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
