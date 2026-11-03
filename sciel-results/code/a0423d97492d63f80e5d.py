"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 120.0          # hard kill from evaluator
_SAFETY = 12.0             # safety margin -> work until ~108s
_START = time.time()

# Precompute upper triangle indices for N=100
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Returns (energy, gradient) where gradient has shape (N, 3).
    The gradient is projected onto the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise squared distances directly (avoid sqrt for speed)
    # dist_sq_ij = |x_i|^2 + |x_j|^2 - 2*x_i.x_j = 2 - 2*x_i.x_j (since on unit sphere)
    dot_products = pts @ pts.T  # (n, n)
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    
    # Clip to avoid negative values from floating point
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    # Energy: -sum_{i<j} log(sqrt(dist_sq_ij)) = -0.5 * sum_{i<j} log(dist_sq_ij)
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    # Weight matrix: w_ij = 1/dist_sq_ij for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_j - x_i) = (W @ pts)_i - W[i,:].sum() * x_i
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return float(energy), grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    return float(-0.5 * np.sum(np.log(dist_sq[_IU])))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    d_upper = dists[iu]
    if np.any(d_upper < 1e-9):
        return float('inf')
    terms = [math.log(1.0 / d) for d in d_upper]
    return math.fsum(terms)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    # min pairwise distance
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from a starting point, respecting deadline."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        # Normalize to sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        
        e, g = _fast_energy_and_grad(p)
        g_flat = g.reshape(-1)
        return e, g_flat
    
    # Phase 1: coarse optimization with moderate tolerances
    if time.time() < deadline - 5.0:
        try:
            x0 = pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxfun': 1000,
                    'ftol': 1e-9,
                    'gtol': 1e-5,
                    'maxls': 15,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: medium refinement
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 400,
                    'maxfun': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 25,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: fine refinement
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 600,
                    'maxfun': 4000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 35,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 108.0  # 12s safety margin before 120s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations (curated to ~8 best starts)
    initializations = []
    
    # Fibonacci spiral with best offsets
    for offset in [0.5, 0.3, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(3):
        amp = 0.03 + 0.02 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points
    for s in range(2):
        initializations.append(_random_sphere(100 + s * 13))
    
    # Golden ratio spiral
    i = np.arange(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / N
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    theta = golden_angle * i
    golden_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(_project(golden_pts))
    
    best_pts = None
    best_energy = float('inf')
    
    # Allocate 60s for multi-restart, 48s for final polish
    restart_budget = 60.0
    restart_deadline = start_time + restart_budget
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > restart_deadline:
            break
        
        remaining = restart_deadline - time.time()
        if remaining < 6.0:
            break
        
        # Give ~7s per restart
        alloc = min(7.0, remaining - 2.0)
        if alloc < 3.0:
            break
        
        rd = time.time() + alloc
        
        optimized, e_precise = _optimize_from_start(init_pts, rd)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Final high-precision polish on the global best (gets ~48s)
    if best_pts is not None and time.time() < deadline - 8.0:
        n = best_pts.shape[0]
        
        def objective_and_grad(flat):
            p = flat.reshape(n, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            p = p / norms
            e, g = _fast_energy_and_grad(p)
            g_flat = g.reshape(-1)
            return e, g_flat
        
        # Multiple polish passes with increasing precision
        polish_deadline = deadline - 8.0
        for polish_pass in range(6):
            if time.time() >= polish_deadline:
                break
            try:
                x0 = best_pts.reshape(-1)
                maxiter = 400 + 300 * polish_pass
                ftol = max(1e-16, 1e-13 - polish_pass * 1e-14)
                gtol = max(1e-12, 1e-9 - polish_pass * 2e-10)
                
                result = minimize(
                    objective_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': maxiter,
                        'maxfun': 4000 + 1500 * polish_pass,
                        'ftol': ftol,
                        'gtol': gtol,
                        'maxls': 50 + 10 * polish_pass,
                    }
                )
                new_pts = _project(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = new_pts
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
