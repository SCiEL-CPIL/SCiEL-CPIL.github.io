"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized ops."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    mask = ~np.eye(N, dtype=bool)
    log_dist = np.log(dist)
    energy = -0.5 * np.sum(log_dist[mask])
    # grad[i] = sum_{j != i} (xi - xj) / |xi - xj|^2
    grad = np.einsum('ijk,ij->ik', diff, 1.0 / dist_sq)
    return energy, grad


def _energy_fsum(pts):
    """Compute energy using math.fsum for accuracy."""
    total = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            total.append(math.log(1.0 / d))
    return math.fsum(total)


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pts)


def _random_points(seed=None):
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, 3)
    return _project_to_sphere(pts)


def _adam_optimize(pts, max_steps=2000, t0=None, deadline=None):
    """Adam optimizer on the sphere."""
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = t0 + 120.0
    
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.01
    step = 0
    
    while time.time() - t0 < deadline:
        energy, grad = _energy_and_grad(pts)
        step += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        pts_new = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
        pts_new = _project_to_sphere(pts_new)
        
        if not np.all(np.isfinite(pts_new)):
            lr *= 0.5
            continue
        
        diff = pts_new[:, np.newaxis, :] - pts_new[np.newaxis, :, :]
        dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        np.fill_diagonal(dist, 1.0)
        if dist.min() < 1e-9:
            lr *= 0.5
            continue
        
        energy_new, _ = _energy_and_grad(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
        pts = pts_new
    
    return best_pts, best_energy


def _lbfgs_polish(pts, t0=None, deadline=None):
    """Polish using L-BFGS-B in tangent space via scipy."""
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = t0 + 120.0
    
    # Use scipy's L-BFGS-B with a parameterization that stays on the sphere
    # by optimizing in R^3 and projecting (simple but effective for local polish)
    def obj_and_grad(x_flat):
        if time.time() - t0 > deadline:
            return 0.0, np.zeros_like(x_flat)
        x = x_flat.reshape(N, 3)
        x = _project_to_sphere(x)
        e, g = _energy_and_grad(x)
        return e, g.flatten()
    
    x0 = pts.flatten()
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12,
                                'maxfun': 1000})
        if res.success or time.time() - t0 > deadline - 5:
            x_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt, _ = _energy_and_grad(x_opt)
            return x_opt, e_opt
    except Exception:
        pass
    return pts, _energy_and_grad(pts)[0]


def _basin_hopping(pts, t0=None, deadline=None):
    """Basin hopping: perturb, then polish with Adam."""
    if t0 is None:
        t0 = time.time()
    if deadline is None:
        deadline = t0 + 120.0
    
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(pts)
    rng = np.random.RandomState(42)
    
    perturb_scale = 0.05
    max_iter = 50
    
    for i in range(max_iter):
        if time.time() - t0 > deadline - 10:
            break
        
        # Perturb
        pts_pert = best_pts + perturb_scale * rng.randn(N, 3)
        pts_pert = _project_to_sphere(pts_pert)
        
        # Quick Adam polish
        pts_polished, e_polished = _adam_optimize(pts_pert, max_steps=200, t0=t0, deadline=deadline)
        
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished.copy()
            perturb_scale = max(0.01, perturb_scale * 0.95)
        else:
            perturb_scale = min(0.2, perturb_scale * 1.1)
    
    return best_pts, best_energy


def construct():
    t0 = time.time()
    deadline = 110.0  # safety margin before 150s limit
    
    # Multiple diverse starts
    starts = []
    starts.append(_fibonacci_sphere())
    starts.append(_random_points(seed=1))
    starts.append(_random_points(seed=42))
    
    # Rotate Fibonacci
    rng = np.random.RandomState(7)
    theta = rng.uniform(0, 2 * np.pi, 3)
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
    Ry = np.array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    R = Ry @ Rx
    starts.append(_project_to_sphere(starts[0] @ R.T))
    
    best_pts = starts[0]
    best_energy = _energy_and_grad(starts[0])[0]
    
    for start in starts[1:]:
        if time.time() - t0 > deadline - 20:
            break
        e, _ = _energy_and_grad(start)
        if e < best_energy:
            best_energy = e
            best_pts = start
    
    # Adam optimization from best start
    best_pts, best_energy = _adam_optimize(best_pts, max_steps=1500, t0=t0, deadline=deadline - 30)
    
    # Basin hopping for escape from local minima
    if time.time() - t0 < deadline - 15:
        best_pts, best_energy = _basin_hopping(best_pts, t0=t0, deadline=deadline - 10)
    
    # L-BFGS polish for high precision
    if time.time() - t0 < deadline - 5:
        best_pts, best_energy = _lbfgs_polish(best_pts, t0=t0, deadline=deadline - 2)
    
    # Final validation and normalization
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(dist, 1.0)
    if dist.min() < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere()
    
    return best_pts
# EVOLVE-BLOCK-END
