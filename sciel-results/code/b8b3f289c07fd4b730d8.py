"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


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
    
    # grad[i] = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(d2)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2[~np.eye(n, dtype=bool)] = 1.0 / d2[~np.eye(n, dtype=bool)]
    
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    
    # Project to tangent space
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


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_from_start(pts0, deadline, max_iter=3000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.02
    momentum = 0.9
    patience = 0
    best_lr = lr
    
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
        if gnorm < 1e-13:
            break
        
        if patience > 50:
            lr = max(lr * 0.5, 1e-8)
            v *= 0.5
            patience = 0
            if lr < 1e-9:
                break
        
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            if e_new < best_energy:
                best_lr = lr
                lr = min(lr * 1.03, 0.15)
        else:
            lr = max(lr * 0.6, 1e-8)
            v *= 0.3
    
    return best_pts, best_energy


def _refine_with_small_steps(pts, deadline, max_iter=1500):
    """Final refinement with very small steps."""
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    lr = 0.002
    v = np.zeros_like(pts)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-13:
            break
        
        v = 0.85 * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        e_new = _log_energy(pts_new)
        
        if e_new < e:
            pts = pts_new
            lr = min(lr * 1.02, 0.01)
        else:
            lr *= 0.5
            v *= 0.5
            if lr < 1e-10:
                break
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_energy = np.inf
    
    # Multiple starting configurations
    starts = []
    
    # Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # Several random starts with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        starts.append(_random_sphere(N, seed=seed))
    
    # Optimize from each start
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 15:
            break
        
        # Give more time to earlier starts, less to later ones
        time_per_start = min(25.0, (deadline - 15 - time.time()) / max(1, len(starts) - i))
        local_deadline = min(deadline - 15, time.time() + time_per_start)
        
        pts_opt, e_opt = _optimize_from_start(pts0, local_deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    # Final refinement
    if best_pts is not None and time.time() < deadline - 3:
        pts_refined, e_refined = _refine_with_small_steps(best_pts, deadline)
        if e_refined < best_energy:
            best_energy = e_refined
            best_pts = pts_refined
    
    return best_pts
# EVOLVE-BLOCK-END
