"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist) = -sum over i<j of log(dist)
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Gradient: for each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dist_sq[i,j]
    # diff[i,j] = x_i - x_j
    grad = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _golden_spiral(offset=0.0):
    """Generate golden-angle spiral points on unit sphere with optional offset."""
    i = np.arange(N) + 0.5 + offset
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts0, deadline):
    """Run optimization from a starting configuration, respecting deadline."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    
    # Use L-BFGS-B with constraints handled by re-projection
    # We'll use a custom approach: optimize in unconstrained space with penalty
    
    best_energy, best_pts = _energy_and_grad(pts)
    
    # Try multiple optimization strategies
    
    # Strategy 1: Direct optimization with scipy minimize (L-BFGS-B)
    # Flatten points and use bounds to keep them reasonable, then project
    x0 = pts.ravel()
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def jac(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        p = x.reshape(-1, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.ravel()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', jac=jac,
                         options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                                  'maxfun': 5000, 'disp': False})
        if result.x is not None and np.isfinite(result.fun):
            pts_new = _project_to_sphere(result.x.reshape(-1, 3))
            e_new, _ = _energy_and_grad(pts_new)
            if e_new < best_energy:
                best_energy, best_pts = e_new, pts_new
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Strategy 2: Iterative refinement with projected gradient
    # Use a simple projected gradient descent with adaptive step size
    pts = best_pts.copy()
    lr = 0.1
    for step in range(5000):
        if time.time() > deadline:
            break
        e, g = _energy_and_grad(pts)
        if e < best_energy:
            best_energy, best_pts = e, pts.copy()
        
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        g_norm = np.linalg.norm(g_tangent)
        if g_norm < 1e-12:
            break
        
        # Take step
        pts_new = pts - lr * g_tangent
        pts_new = _project_to_sphere(pts_new)
        
        # Check if energy improved
        e_new, _ = _energy_and_grad(pts_new)
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.05, 1.0)
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting configurations
    starts = []
    
    # Golden spiral with various offsets
    for offset in [0.0, 0.1, 0.25, 0.5, 0.75]:
        starts.append(_golden_spiral(offset))
    
    # Random starts
    for _ in range(5):
        starts.append(_random_points(rng))
    
    # Best overall
    best_energy = np.inf
    best_pts = starts[0].copy()
    
    # Optimize from each start
    for i, pts0 in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Allocate remaining time evenly among remaining starts
        remaining = len(starts) - i
        time_per_start = max(5.0, (deadline - time.time()) / remaining)
        local_deadline = min(deadline, time.time() + time_per_start)
        
        pts, energy = _optimize_from_start(pts0, local_deadline)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final refinement with more iterations if time permits
    if time.time() < deadline - 10.0:
        pts, energy = _optimize_from_start(best_pts, deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Validate
    if not np.all(np.isfinite(best_pts)):
        # Fallback to golden spiral
        best_pts = _golden_spiral(0.0)
    
    return best_pts
# EVOLVE-BLOCK-END
