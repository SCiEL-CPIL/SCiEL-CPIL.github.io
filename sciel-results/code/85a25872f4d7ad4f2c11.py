"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


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


def _icosahedron_sphere(n):
    """Generate points based on icosahedron symmetry."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = []
    for i in range(n):
        v = verts[i % len(verts)]
        noise = np.random.RandomState(i).randn(3) * 0.1
        v = v + noise
        v /= np.linalg.norm(v)
        pts.append(v)
    return np.array(pts)


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _optimize_lbfgs(pts0, deadline):
    """Optimize using L-BFGS-B with sphere projection."""
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
                          options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40})
        
        if result.success or result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _log_energy(best_pts)
            return best_pts, best_energy
        else:
            return pts0, _log_energy(pts0)
    except Exception:
        return pts0, _log_energy(pts0)


def _optimize_momentum(pts0, deadline, max_iter=5000):
    """Optimize using gradient descent with momentum and adaptive step size."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.01
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
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            if e_new < best_energy:
                best_lr = lr
                lr = min(lr * 1.01, 0.05)
        else:
            lr = max(lr * 0.75, 1e-10)
            v *= 0.5
    
    return best_pts, best_energy


def _polish_high_precision(pts0, deadline, max_iter=2000):
    """Final high-precision polish with very small steps."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    v = np.zeros_like(pts)
    lr = 0.0005
    momentum = 0.85
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _log_energy_grad(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        v = momentum * v - lr * grad
        pts_new = pts + v
        pts_new = _project_to_sphere(pts_new)
        
        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
        else:
            lr *= 0.5
            v *= 0.3
            if lr < 1e-14:
                break
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_pts = None
    best_energy = np.inf
    
    # Multiple diverse starting configurations
    starts = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("random_42", _random_sphere(N, seed=42)),
        ("random_123", _random_sphere(N, seed=123)),
        ("random_999", _random_sphere(N, seed=999)),
        ("random_777", _random_sphere(N, seed=777)),
        ("icosahedron", _icosahedron_sphere(N)),
    ]
    
    for name, pts0 in starts:
        if time.time() > deadline - 20:
            break
        
        # First try L-BFGS-B
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
        
        # Then refine with momentum method
        if time.time() < deadline - 15:
            pts_ref, e_ref = _optimize_momentum(pts_opt, deadline)
            if e_ref < best_energy:
                best_energy = e_ref
                best_pts = pts_ref
    
    # Final high-precision polish pass
    if best_pts is not None and time.time() < deadline - 5:
        pts_polished, e_polished = _polish_high_precision(best_pts, deadline)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished
    
    return best_pts
# EVOLVE-BLOCK-END
