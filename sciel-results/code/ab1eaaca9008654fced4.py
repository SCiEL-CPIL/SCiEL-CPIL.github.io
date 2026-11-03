"""Logarithmic energy minimization for N points on the unit sphere.
Uses multi-start optimization with Sobol initializations and L-BFGS-B,
combining low-discrepancy seeding with robust line-search refinement."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0  # safety margin from 150s limit
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


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
    """Compute energy and gradient (projected onto tangent space).
    Returns (energy, gradient) where gradient is the true gradient of the energy."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = sum log(1/d) = -sum log(d)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # raw_grad is sum (x_i - x_j)/d^2
    # True gradient is -raw_grad
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
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
    
    # Map to sphere
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _optimize(pts, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50):
    """Run L-BFGS-B optimization from a starting point."""
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
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': maxls,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _riemannian_perturb(pts, scale=0.05, rng=None):
    """Perturb points in tangent space to escape local minima."""
    if rng is None:
        rng = np.random.default_rng()
    n = pts.shape[0]
    # Generate random tangent vectors
    random_vecs = rng.standard_normal((n, 3))
    # Project onto tangent space
    dot = np.sum(random_vecs * pts, axis=1, keepdims=True)
    tangent_vecs = random_vecs - dot * pts
    # Normalize
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    tangent_vecs = tangent_vecs / norms
    # Apply perturbation
    perturbed = pts + scale * tangent_vecs
    return _project(perturbed)


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


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy) - better space coverage
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.03 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(10):
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
    
    # Optimize top candidates - use more restarts with L-BFGS-B
    top_candidates = scored_inits[:30]
    
    for idx, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 15.0:
            break
        
        # Phase 1: Coarse optimization
        optimized = _optimize(init_pts, deadline, max_iter=400, ftol=1e-10, gtol=1e-6, maxls=30)
        
        # Phase 2: Fine optimization
        if time.time() < deadline - 8.0:
            optimized = _optimize(optimized, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
        
        # Riemannian perturbation to escape local minima for top candidates
        if idx < 5 and time.time() < deadline - 10.0:
            for perturb_scale in [0.02, 0.05, 0.1]:
                if time.time() > deadline - 6.0:
                    break
                perturbed = _riemannian_perturb(optimized, scale=perturb_scale, rng=rng)
                perturbed_opt = _optimize(perturbed, deadline, max_iter=1500, ftol=1e-14, gtol=1e-11, maxls=50)
                e_pert = _precise_energy(perturbed_opt)
                if e_pert < best_energy:
                    best_energy = e_pert
                    best_pts = perturbed_opt.copy()
            
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 3.0:
        refined = _optimize(best_pts, deadline, max_iter=3000, ftol=1e-16, gtol=1e-13, maxls=50)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
        
        # Second polish with perturbation
        if time.time() < deadline - 1.5:
            for perturb_scale in [0.01, 0.03]:
                if time.time() > deadline - 0.5:
                    break
                perturbed = _riemannian_perturb(best_pts, scale=perturb_scale, rng=rng)
                perturbed_opt = _optimize(perturbed, deadline, max_iter=2000, ftol=1e-16, gtol=1e-14, maxls=50)
                e_pert = _precise_energy(perturbed_opt)
                if e_pert < best_energy:
                    best_energy = e_pert
                    best_pts = perturbed_opt.copy()
    
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
