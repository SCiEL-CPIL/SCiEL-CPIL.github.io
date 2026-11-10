"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy(pts):
    """Compute logarithmic energy."""
    n = len(pts)
    # Compute pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0) # Avoid log(0)
    dist = np.sqrt(dist_sq)
    
    # Mask lower triangle for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_dists = dist[mask]
    
    # Sum log(1/d) = -sum(log(d))
    # Use fsum for precision if needed, but np.sum is usually fine for this scale
    # However, the task mentions math.fsum for the score. We'll stick to np.sum for speed in optimization.
    energy = -np.sum(np.log(valid_dists))
    return energy

def _energy_grad(pts):
    """Compute gradient of logarithmic energy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient of -log(d) w.r.t x_i is (x_i - x_j) / d^2
    # Sum over j != i
    # Term is diff / dist_sq[..., np.newaxis]
    grad_terms = diff / dist_sq[..., np.newaxis]
    grad = np.sum(grad_terms, axis=1)
    return grad

def _project_tangent(pts, grad):
    """Project gradient onto tangent space of sphere at each point."""
    # Tangent space: v . x = 0
    # v_tan = v - (v . x) x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts

def _optimize_from_start(start_pts):
    """Run L-BFGS-B optimization starting from start_pts."""
    # Flatten points to 1D array
    x0 = start_pts.flatten()
    
    def fun_and_grad(x):
        pts = x.reshape(-1, 3)
        # Normalize points to unit sphere to handle numerical drift
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        e = _energy(pts)
        g = _energy_grad(pts)
        g_tan = _project_tangent(pts, g)
        return e, g_tan.flatten()
    
    # Use L-BFGS-B with tight tolerances
    res = minimize(
        fun_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
    )
    
    # Extract and normalize final points
    best_pts = res.x.reshape(-1, 3)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_pts = best_pts / norms
    
    return best_pts, res.fun

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _halton_sphere(n, seed=12345):
    # Simple Halton sequence generator for 3 dimensions
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, n + 1):
        # Use different bases for x, y, z to get 3D points
        # Standard Halton uses primes 2, 3, 5
        u1 = halton(i, 2)
        u2 = halton(i, 3)
        u3 = halton(i, 5)
        
        # Map [0,1]^3 to sphere using spherical coordinates
        # u1 -> theta, u2 -> phi, u3 not used directly or for jitter?
        # Better: use u1, u2 for spherical, and maybe a 3rd dimension for variation or just use 2D Halton mapped to sphere
        # Actually, for sphere, we can use u1 for z, u2 for phi
        z = 1.0 - 2.0 * u1
        phi = 2.0 * np.pi * u2
        r = np.sqrt(1.0 - z * z)
        points.append([r * np.cos(phi), r * np.sin(phi), z])
    return np.array(points)

def _random_sphere(n, seed=0):
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    best_pts = None
    best_energy = float('inf')
    
    # List of initializers
    initializers = [
        _fibonacci_sphere,
        _halton_sphere,
        lambda n: _random_sphere(n, seed=1),
        lambda n: _random_sphere(n, seed=2),
        lambda n: _random_sphere(n, seed=3),
        lambda n: _random_sphere(n, seed=4),
        lambda n: _random_sphere(n, seed=5),
        lambda n: _random_sphere(n, seed=6),
        lambda n: _random_sphere(n, seed=7),
        lambda n: _random_sphere(n, seed=8),
    ]
    
    for init_func in initializers:
        if time.time() > deadline:
            break
            
        try:
            start_pts = init_func(N)
            # Ensure start points are valid
            if not np.all(np.isfinite(start_pts)):
                continue
            
            pts, energy = _optimize_from_start(start_pts)
            
            # Check validity
            if not np.all(np.isfinite(pts)):
                continue
            
            # Check min distance
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, 1.0)
            if np.min(dists) < 1e-9:
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
                
        except Exception:
            continue
            
    if best_pts is None:
        # Fallback to Fibonacci if all failed
        best_pts = _fibonacci_sphere(N)
        
    return best_pts

# EVOLVE-BLOCK-END
