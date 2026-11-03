"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

# Precompute upper triangle indices for N=100
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations."""
    n = pts.shape[0]
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad = W @ pts - W_row_sum * pts
    
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


def _fast_energy_precise(pts):
    """Compute energy with fsum over vectorized distances (much faster than per-pair Python loop)."""
    n = pts.shape[0]
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dists = np.sqrt(dist_sq[_IU])
    if np.any(dists < 1e-9):
        return float('inf')
    return math.fsum(np.log(1.0 / dists).tolist())


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


def _sobol_sphere(seed):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(N)
    samples = samples - 0.5
    return _project(samples)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """Gradient flow with adaptive learning rate and annealing."""
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy_only(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _fast_energy_and_grad(current)
        
        # Adaptive learning rate: scale by inverse gradient magnitude
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-12:
            lr = min(lr, 0.3 / g_norm)
        
        # Add small noise for exploration (annealed)
        noise_scale = 0.003 * (1.0 - progress) ** 1.5
        noise = rng.standard_normal((n, 3)) * noise_scale
        
        # Update in ambient space, then project back to sphere
        current = current - lr * g + noise
        current = _project(current)
        
        e_new = _fast_energy_only(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy


def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from a starting point, respecting deadline."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _fast_energy_precise(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: coarse optimization
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective_and_grad,
                pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 2500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_new = _fast_energy_only(new_pts)
            if e_new < best_energy:
                best_pts = new_pts
                best_energy = _fast_energy_precise(new_pts)
        except Exception:
            pass
    
    # Phase 2: medium refinement
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'maxfun': 4000, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 30}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_new = _fast_energy_only(new_pts)
            if e_new < best_energy:
                best_pts = new_pts
                best_energy = _fast_energy_precise(new_pts)
        except Exception:
            pass
    
    # Phase 3: fine refinement
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1500, 'maxfun': 8000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_new = _fast_energy_only(new_pts)
            if e_new < best_energy:
                best_pts = new_pts
                best_energy = _fast_energy_precise(new_pts)
        except Exception:
            pass
    
    return best_pts, best_energy


def _final_polish(pts, deadline):
    """Final ultra-precise polish with tiny perturbations and multiple restarts."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _fast_energy_precise(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    rng = np.random.default_rng(12345)
    # Perturbations: focus on smaller scales for fine-tuning
    perturbations = [0.0, 3e-4, 1.5e-4, 7e-5, 3e-5, 1.5e-5, 7e-6, 3e-6, 1e-6, 5e-7]
    
    for scale in perturbations:
        if time.time() > deadline - 3.0:
            break
        
        # Try multiple restarts for each scale
        n_restarts = 3 if 0 < scale <= 1e-4 else (2 if scale > 0 else 1)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 3.0:
                break
            
            if scale == 0.0:
                start = best_pts
            else:
                start = _project(best_pts + rng.standard_normal((n, 3)) * scale)
            
            try:
                result = minimize(
                    objective_and_grad,
                    start.reshape(-1),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 40000,
                        'maxfun': 200000,
                        'ftol': 1e-18,
                        'gtol': 1e-16,
                        'maxls': 60,
                    }
                )
                new_pts = _project(result.x.reshape(n, 3))
                e_new = _fast_energy_only(new_pts)
                if e_new < best_energy:
                    e_precise = _fast_energy_precise(new_pts)
                    if e_precise < best_energy:
                        best_pts = new_pts
                        best_energy = e_precise
            except Exception:
                pass
    
    # Final Riemannian gradient descent polish with very small steps
    if time.time() < deadline - 1.0:
        pts = best_pts.copy()
        for step_iter in range(1500):
            if time.time() > deadline - 1.0:
                break
            e, g = _fast_energy_and_grad(pts)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-15:
                break
            lr = min(0.006 / (1.0 + 0.008 * step_iter), 0.3 / g_norm)
            pts_new = _project(pts - lr * g)
            e_new = _fast_energy_only(pts_new)
            if e_new < e:
                pts = pts_new
            else:
                lr *= 0.5
                pts_new = _project(pts - lr * g)
                e_new = _fast_energy_only(pts_new)
                if e_new < e:
                    pts = pts_new
                else:
                    break
        
        e_final = _fast_energy_precise(pts)
        if e_final < best_energy:
            best_pts = pts
            best_energy = e_final
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    for offset in [0.5, 0.3, 0.7, 0.2, 0.8]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(5):
        amp = 0.015 + 0.015 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(6):
        initializations.append(_random_sphere(100 + s * 17))
    
    # Sobol sequences - better uniform coverage
    for seed in [1, 2, 3, 4]:
        initializations.append(_sobol_sphere(seed))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Quick screen with gradient flow + short optimization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 45.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
        
        # Gradient flow for exploration
        alloc_flow = min(5.0, remaining - 18.0)
        if alloc_flow < 1.5:
            break
        
        flow_deadline = time.time() + alloc_flow
        try:
            flow_pts, flow_energy = _gradient_flow(
                init_pts, rng, 
                n_steps=120, 
                lr_start=0.04, 
                lr_end=0.001,
                deadline=flow_deadline
            )
            init_pts = flow_pts
        except Exception:
            pass
        
        # Short optimization: just enough to identify the basin
        alloc_opt = min(6.0, remaining - 15.0)
        if alloc_opt < 2.5:
            break
        
        restart_deadline = time.time() + alloc_opt
        
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
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10, 'gtol': 1e-5, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(N, 3))
            e_new = _fast_energy_only(new_pts)
            if e_new < best_energy:
                best_energy = _fast_energy_precise(new_pts)
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase B: Deep optimization from the best found so far
    if best_pts is not None and time.time() < deadline - 30.0:
        remaining = deadline - time.time()
        deep_deadline = time.time() + min(remaining - 25.0, 60.0)
        optimized, e_precise = _optimize_from_start(best_pts, deep_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Phase C: Final high-precision polish with tiny perturbations
    if best_pts is not None and time.time() < deadline - 12.0:
        polish_deadline = deadline - 6.0
        polished, e_precise = _final_polish(best_pts, polish_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    # Phase D: Riemannian gradient descent polish (small steps against gradient)
    if best_pts is not None and time.time() < deadline - 5.0:
        pts = best_pts.copy()
        n = pts.shape[0]
        for step_iter in range(800):
            if time.time() > deadline - 4.0:
                break
            e, g = _fast_energy_and_grad(pts)
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-14:
                break
            lr = min(0.008 / (1.0 + 0.015 * step_iter), 0.3 / g_norm)
            pts_new = _project(pts - lr * g)
            e_new = _fast_energy_only(pts_new)
            if e_new < e:
                pts = pts_new
            else:
                lr *= 0.5
                pts_new = _project(pts - lr * g)
                e_new = _fast_energy_only(pts_new)
                if e_new < e:
                    pts = pts_new
                else:
                    break
        
        e_phase_d = _fast_energy_precise(pts)
        if e_phase_d < best_energy:
            best_energy = e_phase_d
            best_pts = pts
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation using precise energy
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
