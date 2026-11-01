import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
_DEADLINE = 145.0  # 5s safety margin before 150s kill

def _log_energy_fast(pts):
    """Vectorized log energy computation for speed using squared distances."""
    n = len(pts)
    # Using dot products for squared distances is faster
    # d^2 = 2 - 2 * dot(x_i, x_j) for unit vectors
    dots = pts @ pts.T
    dists_sq = 2.0 - 2.0 * dots
    # Numerical stability
    dists_sq = np.maximum(dists_sq, 1e-15)
    
    iu = np.triu_indices(n, k=1)
    d2 = dists_sq[iu]
    
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * np.sum(np.log(d2))

def _halton_sphere(n, base=2):
    """Generate Halton sequence points on sphere (low-discrepancy)."""
    rng = np.random.default_rng(42)
    pts = np.zeros((n, 3))
    for i in range(n):
        # Use Halton sequence in 2D, map to sphere via spherical coords
        u = _halton_value(i + 1, 2)
        v = _halton_value(i + 1, 3)
        theta = 2.0 * math.pi * u
        phi = math.acos(1.0 - 2.0 * v)
        pts[i, 0] = math.sin(phi) * math.cos(theta)
        pts[i, 1] = math.sin(phi) * math.sin(theta)
        pts[i, 2] = math.cos(phi)
    return pts

def _halton_value(index, base):
    """Compute Halton sequence value for given index and base."""
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def _log_energy_grad(pts):
    """Compute gradient of logarithmic energy w.r.t. points on sphere (projected)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i, k] = sum_j (-(pts[i,k] - pts[j,k]) / dists_sq[i,j])
    # = -pts[i,k] * sum_j(1/d2) + sum_j(pts[j,k] / d2)
    weighted_sum_pts = inv_d2 @ pts
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -pts * sum_inv_d2[:, np.newaxis] + weighted_sum_pts
    
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    return grad

def _energy_and_grad_3d(x):
    """Energy and gradient for L-BFGS-B on 3D coordinates."""
    pts = x.reshape(N, 3)
    energy = _log_energy_fast(pts)
    grad = _log_energy_grad(pts)
    return energy, grad.flatten()

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _exponential_map(pts, delta_pts):
    """Exponential map on sphere: move points along tangent vectors."""
    norms = np.linalg.norm(delta_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    sin_norm = np.sin(norms)
    cos_norm = np.cos(norms)
    # exp_map(x, v) = x*cos(||v||) + (v/||v||)*sin(||v||)
    new_pts = pts * cos_norm + delta_pts * (sin_norm / norms)
    return new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

def _retract(pts, delta_pts):
    """Retraction on sphere: simple normalization."""
    new_pts = pts + delta_pts
    return new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

def _riemannian_cg(pts, max_iter=500, tol=1e-12):
    """Riemannian conjugate gradient on the sphere manifold."""
    n = len(pts)
    pts = _normalize(pts.copy())
    grad = _log_energy_grad(pts)
    # Project gradient to tangent space
    dot_prod = np.sum(grad * pts, axis=1)
    grad -= dot_prod[:, np.newaxis] * pts
    
    y = -grad
    p = y.copy()
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-15:
        return pts
    
    prev_grad_norm = np.linalg.norm(grad)
    energy = _log_energy_fast(pts)
    
    for it in range(max_iter):
        # Line search (backtracking)
        step = 0.1
        new_pts = _retract(pts, step * p)
        new_energy = _log_energy_fast(new_pts)
        
        while new_energy > energy and step > 1e-8:
            step *= 0.5
            new_pts = _retract(pts, step * p)
            new_energy = _log_energy_fast(new_pts)
        
        if step <= 1e-8:
            break
        
        pts = new_pts
        energy = new_energy
        
        new_grad = _log_energy_grad(pts)
        dot_prod = np.sum(new_grad * pts, axis=1)
        new_grad -= dot_prod[:, np.newaxis] * pts
        new_grad_norm = np.linalg.norm(new_grad)
        
        if new_grad_norm < tol:
            break
        
        # Fletcher-Reeves update
        alpha = np.sum(grad * new_grad) / max(np.sum(grad * grad), 1e-15)
        y = -new_grad
        p = y + alpha * p
        
        grad = new_grad
        prev_grad_norm = new_grad_norm
    
    return pts

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_energy = float('inf')
    best_pts = None
    
    # Generate initializations
    initializations = []
    
    # 1. Fibonacci Sphere
    fib_pts = _fibonacci_sphere(N)
    initializations.append(fib_pts.copy())
    
    # 2. Halton Sphere (low-discrepancy)
    halton_pts = _halton_sphere(N)
    initializations.append(halton_pts.copy())
    
    # 3. Random uniform on sphere
    for seed in range(4):
        rng_local = np.random.default_rng(seed * 123 + 7)
        pts_rand = rng_local.standard_normal((N, 3))
        pts_rand = _normalize(pts_rand)
        initializations.append(pts_rand)
    
    # 4. Perturbed Fibonacci
    for seed in range(3):
        rng_local = np.random.default_rng(seed * 456 + 13)
        pts_pert = fib_pts.copy() + rng_local.standard_normal((N, 3)) * 0.02
        pts_pert = _normalize(pts_pert)
        initializations.append(pts_pert)
    
    # 5. Slightly different Fibonacci variants
    for offset in [0.3, 0.7]:
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        initializations.append(_normalize(pts))
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # Phase 1: L-BFGS-B on 3D coordinates (fast coarse optimization)
        x0 = init_pts.flatten()
        
        try:
            res = minimize(
                _energy_and_grad_3d,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 3000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            current_pts = _normalize(res.x.reshape(N, 3))
            current_energy = _log_energy_fast(current_pts)
            
            # Phase 2: Riemannian CG polish (stays on sphere, tighter convergence)
            if time.time() < deadline - 10:
                rc_pts = _riemannian_cg(current_pts, max_iter=300, tol=1e-13)
                rc_energy = _log_energy_fast(rc_pts)
                if rc_energy < current_energy:
                    current_pts = rc_pts
                    current_energy = rc_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
                
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = initializations[0].copy()
        best_energy = _log_energy_fast(best_pts)
    
    # Final high-precision polish
    if time.time() < deadline - 3:
        current_pts = best_pts.copy()
        current_energy = best_energy
        
        # Riemannian CG with very tight tolerance
        try:
            polished_pts = _riemannian_cg(current_pts, max_iter=1000, tol=1e-14)
            polished_energy = _log_energy_fast(polished_pts)
            
            if polished_energy < current_energy:
                best_energy = polished_energy
                best_pts = polished_pts
        except Exception:
            pass
        
        # Additional L-BFGS-B polish if time remains
        if time.time() < deadline - 1:
            try:
                x0 = best_pts.flatten()
                res = minimize(
                    _energy_and_grad_3d,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-16,
                        'gtol': 1e-13,
                        'maxls': 100
                    }
                )
                polished_pts = _normalize(res.x.reshape(N, 3))
                polished_energy = _log_energy_fast(polished_pts)
                
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_pts = polished_pts
            except Exception:
                pass
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    return best_pts
