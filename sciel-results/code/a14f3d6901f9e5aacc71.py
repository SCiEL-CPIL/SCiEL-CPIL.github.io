"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start BFGS optimization with Nelder-Mead refinement and diverse initializations."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_START = time.time()


def _remaining(deadline):
    return deadline - time.time()


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient projected onto tangent space."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    grad = -raw_grad
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


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


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _quick_grad_descent(pts, n_steps=50, lr=0.01):
    """Quick gradient descent to improve starting point."""
    current = pts.copy()
    best_e = _energy(current)
    for step in range(n_steps):
        e, g = _energy_grad(current)
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        step_lr = lr * (1.0 / (1.0 + gnorm))
        new_pts = _project(current - step_lr * g)
        new_e = _energy(new_pts)
        if new_e < best_e:
            current = new_pts
            best_e = new_e
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
    return current


def _bfgs_optimize(pts, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run BFGS optimization (full Hessian approximation) from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='BFGS',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _nelder_mead_refine(pts, max_iter=300, tol=1e-13):
    """Nelder-Mead local refinement for final polish."""
    n = pts.shape[0]
    
    def objective(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        return _energy(p)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin from 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - 15 seeds
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals - 8 variants
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds - 12 seeds
    for s in range(12):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates with BFGS (structural change: full Hessian)
    top_candidates = scored_inits[:40]
    
    for idx, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 20.0:
            break
        
        # Phase 1: Quick gradient descent to improve starting point
        pre_pts = _quick_grad_descent(init_pts, n_steps=40, lr=0.02)
        
        # Phase 2: BFGS optimization with moderate tolerances
        optimized = _bfgs_optimize(pre_pts, max_iter=800, ftol=1e-14, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Final multi-phase polish on the global best
    if best_pts is not None and time.time() < deadline - 12.0:
        # Polish 1: BFGS with tight tolerance
        refined = _bfgs_optimize(best_pts, max_iter=1500, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
        
        # Polish 2: Nelder-Mead for local refinement
        if time.time() < deadline - 8.0:
            nm_pts = _nelder_mead_refine(best_pts, max_iter=400, tol=1e-13)
            e_nm = _precise_energy(nm_pts)
            if e_nm < best_energy:
                best_energy = e_nm
                best_pts = nm_pts
        
        # Polish 3: BFGS with ultra-tight tolerance
        if time.time() < deadline - 5.0:
            refined2 = _bfgs_optimize(best_pts, max_iter=2000, ftol=1e-18, gtol=1e-16)
            e_precise = _precise_energy(refined2)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = refined2
        
        # Polish 4: Final Nelder-Mead pass
        if time.time() < deadline - 3.0:
            nm_pts2 = _nelder_mead_refine(best_pts, max_iter=500, tol=1e-14)
            e_nm2 = _precise_energy(nm_pts2)
            if e_nm2 < best_energy:
                best_energy = e_nm2
                best_pts = nm_pts2
    
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
