"""Optimized 100-point log-energy minimization on S^2 with multi-phase strategy."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 5.0  # Safety margin in seconds


def _log_energy(pts):
    """Compute logarithmic energy using math.fsum for accuracy."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-9) or not np.all(np.isfinite(d_upper)):
        return 1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """Compute energy and gradient on sphere."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-12):
        return 1e18, np.zeros_like(pts)
    
    energy = -np.sum(np.log(d_upper))
    
    inv_d2 = np.zeros_like(d2)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]
    
    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts
    
    return energy, grad


def _project(pts):
    """Project to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, offset=0.5):
    """Fibonacci spiral initialization."""
    golden = 0.5 * (1 + np.sqrt(5))
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * golden * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Random points on sphere."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def _optimize_lbfgs(pts0, deadline, maxiter=3000, ftol=1e-15, gtol=1e-12):
    """L-BFGS optimization with sphere projection."""
    x0 = pts0.flatten()
    
    def obj(x):
        pts = _project(x.reshape(-1, 3))
        e, _ = _log_energy_grad(pts)
        return e
    
    def grad(x):
        pts = _project(x.reshape(-1, 3))
        _, g = _log_energy_grad(pts)
        return g.flatten()
    
    try:
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'maxfun': maxiter * 5,
                                  'ftol': ftol, 'gtol': gtol, 'm': 50})
        if result.fun < 1e17:
            pts_opt = _project(result.x.reshape(-1, 3))
            return pts_opt, _log_energy(pts_opt)
    except Exception:
        pass
    return pts0, _log_energy(pts0)


def _gradient_descent(pts0, deadline, max_iter=20000):
    """Gradient descent with momentum and adaptive step."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_e = _log_energy(pts)
    v = np.zeros_like(pts)
    lr = 0.005
    momentum = 0.9
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _log_energy_grad(pts)
        if e < best_e - 1e-16:
            best_e = e
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-15:
            break
        
        v = momentum * v - lr * grad
        pts_new = _project(pts + v)
        e_new = _log_energy(pts_new)
        
        if e_new < e:
            pts = pts_new
            if no_improve < 5:
                lr = min(lr * 1.01, 0.02)
        else:
            lr *= 0.8
            v *= 0.5
            if no_improve > 50:
                v = np.zeros_like(pts)
                lr = 0.005
                no_improve = 0
            if lr < 1e-16:
                break
    
    return best_pts, best_e


def construct():
    """Construct 100 points minimizing log-energy on unit sphere."""
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    best_pts = None
    best_energy = np.inf
    
    # Phase 1: Generate diverse initial configurations
    starts = []
    # Fibonacci with various offsets
    for off in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        starts.append(_fibonacci_sphere(N, offset=off))
    
    # Random starts
    for seed in range(20):
        starts.append(_random_sphere(N, seed=seed))
    
    # Phase 2: Quick optimization on all starts
    candidates = []
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 60:
            break
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline - 60, maxiter=500, ftol=1e-12, gtol=1e-8)
        candidates.append((e_opt, pts_opt))
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
    
    candidates.sort(key=lambda x: x[0])
    
    # Phase 3: Deep optimization on top candidates
    for e_c, pts_c in candidates[:8]:
        if time.time() > deadline - 40:
            break
        pts_deep, e_deep = _optimize_lbfgs(pts_c, deadline - 40, maxiter=2000, ftol=1e-14, gtol=1e-10)
        if e_deep < best_energy:
            best_energy = e_deep
            best_pts = pts_deep
    
    # Phase 4: Gradient descent refinement
    if best_pts is not None and time.time() < deadline - 20:
        pts_gd, e_gd = _gradient_descent(best_pts, deadline - 20, max_iter=15000)
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = pts_gd
    
    # Phase 5: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        pts_final, e_final = _optimize_lbfgs(best_pts, deadline - 2, maxiter=5000, ftol=1e-16, gtol=1e-14)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final
    
    # Ensure validity
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, offset=0.5)
    
    best_pts = _project(best_pts)
    
    # Check for near-duplicates
    min_d = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            min_d = min(min_d, d)
    
    if min_d < 1e-9:
        rng = np.random.RandomState(999)
        best_pts = _project(best_pts + rng.randn(N, 3) * 1e-5)
        if time.time() < deadline - 1:
            best_pts, _ = _optimize_lbfgs(best_pts, deadline - 0.5, maxiter=1000)
    
    return best_pts
