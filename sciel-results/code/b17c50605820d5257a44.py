"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0          # hard kill from evaluator
_SAFETY = 15.0             # safety margin -> work until ~135s
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
    # Compute pairwise squared distances directly
    dot_products = pts @ pts.T  # (n, n)
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    
    # Clip to avoid negative values from floating point
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    # Energy: -sum_{i<j} log(sqrt(dist_sq_ij)) = -0.5 * sum_{i<j} log(dist_sq_ij)
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    # Weight matrix: w_ij = 1/dist_sq_ij for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_j - x_i) = (W @ pts)_i - W[i,:].sum() * x_i
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad = W @ pts - W_row_sum * pts
    
    # Project gradient onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot * pts
    
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
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
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
        # Normalize to sphere (needed since L-BFGS-B explores off-sphere)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: coarse optimization
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: medium refinement
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 2500, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 30}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: fine refinement
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 40}
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
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations (reduced to avoid redundant work)
    initializations = []
    
    # Fibonacci spiral with various offsets (most promising starts)
    for offset in [0.5, 0.3, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(4):
        amp = 0.02 + 0.02 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(5):
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
    
    # Phase A: Quick screen — short optimization from each init to find best basin
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        # Short optimization: just enough to identify the basin
        alloc = min(5.0, remaining - 12.0)
        if alloc < 2.0:
            break
        
        restart_deadline = time.time() + alloc
        
        try:
            def obj(flat):
                p = flat.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                np.maximum(norms, 1e-12, out=norms)
                p /= norms
                e, g = _fast_energy_and_grad(p)
                return e, g.reshape(-1)
            
            result = minimize(
                obj, init_pts.reshape(-1),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-4, 'maxls': 10}
            )
            new_pts = _project(result.x.reshape(N, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase B: Deep optimization from the best found so far
    if best_pts is not None and time.time() < deadline - 20.0:
        remaining = deadline - time.time()
        deep_deadline = time.time() + min(remaining - 15.0, 50.0)
        optimized, e_precise = _optimize_from_start(best_pts, deep_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Phase C: Final high-precision polish with multiple passes
    if best_pts is not None and time.time() < deadline - 8.0:
        n = best_pts.shape[0]
        
        def obj(flat):
            p = flat.reshape(n, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            p /= norms
            e, g = _fast_energy_and_grad(p)
            return e, g.reshape(-1)
        
        polish_configs = [
            {'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40},
            {'maxiter': 800, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50},
            {'maxiter': 1000, 'maxfun': 6000, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 60},
        ]
        
        for cfg in polish_configs:
            if time.time() < deadline - 5.0:
                try:
                    result = minimize(
                        obj, best_pts.reshape(-1),
                        method='L-BFGS-B', jac=True, options=cfg
                    )
                    new_pts = _project(result.x.reshape(n, 3))
                    e_precise = _precise_energy(new_pts)
                    if e_precise < best_energy:
                        best_energy = e_precise
                        best_pts = new_pts
                except Exception:
                    pass
    
    # Phase D: Riemannian Newton-style local polish (small steps along gradient)
    if best_pts is not None and time.time() < deadline - 4.0:
        pts = best_pts.copy()
        n = pts.shape[0]
        # Gradient flow with decreasing step size
        for step_iter in range(200):
            if time.time() > deadline - 3.0:
                break
            e, g = _fast_energy_and_grad(pts)
            # Adaptive step: start moderate, decrease
            lr = 0.01 / (1.0 + 0.05 * step_iter)
            pts_new = pts + lr * g
            pts_new = _project(pts_new)
            e_new = _precise_energy(pts_new)
            if e_new < e:
                pts = pts_new
            else:
                # Halve step
                lr *= 0.5
                pts_new = pts + lr * g
                pts_new = _project(pts_new)
                e_new = _precise_energy(pts_new)
                if e_new < e:
                    pts = pts_new
                else:
                    break  # stuck, stop
        
        if _precise_energy(pts) < best_energy:
            best_energy = _precise_energy(pts)
            best_pts = pts
    
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
