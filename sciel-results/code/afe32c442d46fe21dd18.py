"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
def _energy(x):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # x shape (N, 3)
    # Compute all pairwise distances
    # Use efficient computation
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Avoid self-pairs (diagonal)
    np.fill_diagonal(dist, 1.0)
    # Only sum upper triangle
    iu = np.triu_indices(N, k=1)
    return math.fsum(math.log(1.0 / dist[iu]))


def _energy_and_grad(x):
    """Compute energy and gradient on the sphere (ambient gradient projected)."""
    N = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # dist = sqrt(dist_sq), so log(dist) = 0.5 * log(dist_sq)
    # E = -0.5 * sum_{i<j} log(dist_sq_{ij})
    
    iu = np.triu_indices(N, k=1)
    E = -0.5 * math.fsum(math.log(dist_sq[iu]))
    
    # Gradient of E w.r.t. x:
    # dE/dx_i = sum_{j != i} d/dx_i [-0.5 log(dist_sq_{ij})]
    # = sum_{j != i} -0.5 * (2*(x_i - x_j) / dist_sq_{ij})
    # = -sum_{j != i} (x_i - x_j) / dist_sq_{ij}
    
    # Vectorized: for each i, grad_i = -sum_{j != i} (x_i - x_j) / dist_sq_{ij}
    # (x_i - x_j) / dist_sq_{ij} = diff[i,j] / dist_sq[i,j]
    
    # Build matrix of (x_i - x_j) / dist_sq_{ij} for all pairs
    # diff[i,j] is (x_i - x_j), shape (N, N, 3)
    # dist_sq[i,j] shape (N, N)
    
    # We need to sum over j for each i
    # term[i,j] = diff[i,j] / dist_sq[i,j]  (broadcast)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dist_sq = 1.0 / dist_sq
    inv_dist_sq[np.diag_indices(N)] = 0.0  # exclude self
    
    term = diff * inv_dist_sq[:, :, np.newaxis]  # (N, N, 3)
    # grad_i = -sum_j term[i,j]
    grad = -np.sum(term, axis=1)
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x) * x
    dot = np.sum(grad * x, axis=1, keepdims=True)
    grad_tangent = grad - dot * x
    
    return E, grad_tangent


def _fibonacci_spiral(n, seed=0):
    """Generate n points on sphere using Fibonacci spiral with optional jitter."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if seed > 0:
        # Add small jitter
        jitter = rng.normal(0, 0.01, size=pts.shape)
        pts = pts + jitter
        # Renormalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    return pts


def _optimize_from_start(x0, time_limit, max_iter=5000, tol=1e-12):
    """Optimize using L-BFGS-B in spherical coordinates."""
    from scipy.optimize import minimize
    
    start_time = time.time()
    n = len(x0)
    
    # Convert to spherical coordinates (theta, phi)
    # theta: polar angle [0, pi], phi: azimuthal [0, 2*pi)
    z = x0[:, 2]
    xy = np.sqrt(np.sum(x0[:, :2]**2, axis=1))
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(x0[:, 1], x0[:, 0])
    
    # Parameters: theta (n,), phi (n,)
    # Use a flat parameter vector
    params = np.concatenate([theta, phi])
    
    def obj_and_grad(p):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return float('inf'), None
        
        th = p[:n]
        ph = p[n:]
        
        # Convert to Cartesian
        st = np.sin(th)
        ct = np.cos(th)
        x = st * np.cos(ph)
        y = st * np.sin(ph)
        z = ct
        pts = np.column_stack([x, y, z])
        
        E, grad_ambient = _energy_and_grad(pts)
        
        # Project gradient to spherical coordinates
        # dE/dtheta_i = grad_ambient_i . d(pts_i)/dtheta_i
        # d(pts_i)/dtheta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
        dtheta = np.column_stack([ct * np.cos(ph), ct * np.sin(ph), -st])
        dphi = np.column_stack([-st * np.sin(ph), st * np.cos(ph), np.zeros(n)])
        
        grad_theta = np.sum(grad_ambient * dtheta, axis=1)
        grad_phi = np.sum(grad_ambient * dphi, axis=1)
        
        grad = np.concatenate([grad_theta, grad_phi])
        
        return E, grad
    
    # Bounds for theta and phi
    bounds = [(1e-6, np.pi - 1e-6)] * n + [(0, 2 * np.pi)] * n
    
    best_x = x0.copy()
    best_E = _energy(x0)
    
    try:
        result = minimize(
            obj_and_grad,
            params,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result.success or result.fun < best_E:
            p = result.x
            th = p[:n]
            ph = p[n:]
            st = np.sin(th)
            ct = np.cos(th)
            x = st * np.cos(ph)
            y = st * np.sin(ph)
            z = ct
            new_pts = np.column_stack([x, y, z])
            new_E = _energy(new_pts)
            if new_E < best_E:
                best_x = new_pts
                best_E = new_E
    except Exception:
        pass
    
    return best_x, best_E


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple random starts and keep the best
    best_pts = None
    best_E = float('inf')
    
    # First try the standard Fibonacci spiral
    x0 = _fibonacci_spiral(N, seed=0)
    E0 = _energy(x0)
    best_pts = x0.copy()
    best_E = E0
    
    # Try optimizing from the standard start
    if time.time() < deadline:
        pts, E = _optimize_from_start(x0, deadline - time.time() - 5.0, max_iter=3000)
        if E < best_E:
            best_pts = pts
            best_E = E
    
    # Try a few jittered starts
    for seed in range(1, 10):
        if time.time() > deadline - 15.0:
            break
        
        x0 = _fibonacci_spiral(N, seed=seed)
        E0 = _energy(x0)
        
        if E0 < best_E:
            best_pts = x0.copy()
            best_E = E0
        
        pts, E = _optimize_from_start(x0, deadline - time.time() - 5.0, max_iter=2000)
        if E < best_E:
            best_pts = pts
            best_E = E
    
    # Final polish: run one more L-BFGS-B with tighter tolerance if time permits
    if time.time() < deadline - 10.0 and best_pts is not None:
        pts, E = _optimize_from_start(best_pts, deadline - time.time() - 3.0, max_iter=5000, tol=1e-14)
        if E < best_E:
            best_pts = pts
            best_E = E
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_spiral(N, seed=0)
    
    return best_pts
# EVOLVE-BLOCK-END
