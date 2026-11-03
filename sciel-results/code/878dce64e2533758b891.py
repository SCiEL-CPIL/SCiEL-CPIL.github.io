"""
Logarithmic energy minimization for 100 points on the unit sphere.
Uses multi-restart L-BFGS-B with gradient flow annealing and final high-precision polishing.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 150.0
_SAFETY = 12.0
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
    
    # Project gradient onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot * pts
    
    return float(energy), grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
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


def _sobol_sphere(seed):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    samples = sampler.random(N)
    samples = samples - 0.5
    return _project(samples)


def _repulsive_spread(pts, n_steps=50, lr=0.01, deadline=None):
    """Pre-optimization: spread points using repulsive 1/r^2 energy to avoid clustering."""
    n = pts.shape[0]
    current = pts.copy()
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute repulsive gradient (1/r^2 repulsion)
        dot_products = current @ current.T
        dist_sq = 2.0 - 2.0 * dot_products
        np.fill_diagonal(dist_sq, 1.0)
        np.maximum(dist_sq, 1e-10, out=dist_sq)
        
        # Repulsive force: proportional to 1/dist_sq (gradient of -log is 1/r^2, but we use 1/r^2 repulsion)
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        
        W_row_sum = W.sum(axis=1, keepdims=True)
        grad = W @ current - W_row_sum * current
        
        # Project gradient onto tangent space
        dot = np.sum(grad * current, axis=1, keepdims=True)
        grad -= dot * current
        
        # Update
        current = current - lr * grad
        current = _project(current)
        
        # Anneal learning rate
        lr *= 0.98
    
    return current


def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline, use_noise=True):
    """Gradient flow with adaptive learning rate and optional annealed noise."""
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
            lr = min(lr, 0.5 / g_norm)
        
        # Add small noise for exploration (annealed) only if requested
        if use_noise:
            noise_scale = 0.002 * (1.0 - progress) ** 2.0
            noise = rng.standard_normal((n, 3)) * noise_scale
        else:
            noise = np.zeros((n, 3))
        
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
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: coarse optimization
    if time.time() < deadline - 4.0:
        try:
            result = minimize(
                objective_and_grad,
                pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'maxfun': 1000, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: gradient flow to explore and escape local minima (with noise)
    if time.time() < deadline - 3.0:
        try:
            rng = np.random.default_rng(hash(tuple(np.round(best_pts, 6).flatten())) % (2**32))
            gf_pts, gf_energy = _gradient_flow(
                best_pts, rng, 
                n_steps=80, lr_start=0.008, lr_end=1e-5,
                deadline=deadline - 2.0,
                use_noise=True
            )
            if gf_energy < best_energy:
                best_pts = gf_pts
                best_energy = gf_energy
        except Exception:
            pass
    
    # Phase 3: medium refinement (no noise, pure gradient)
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 400, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 4: fine refinement
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 600, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy


def _final_polish(pts, deadline):
    """Final ultra-precise polish with tiny perturbations and multiple restarts."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    rng = np.random.default_rng(12345)
    # Perturbation scales from medium to tiny
    perturbations = [0.0, 3e-4, 2e-4, 1.5e-4, 1e-4, 7e-5, 5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7]
    
    for scale in perturbations:
        if time.time() > deadline - 1.5:
            break
        
        # Try multiple restarts for each scale to escape shallow local minima
        n_restarts = 2 if scale > 0 else 1
        
        for restart in range(n_restarts):
            if time.time() > deadline - 1.5:
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
                    options={'maxiter': 500, 'maxfun': 2500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
                )
                new_pts = _project(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_energy = e_precise
                    best_pts = new_pts
            except Exception:
                pass
    
    # Final pure gradient descent polish (no noise)
    if time.time() < deadline - 1.0:
        try:
            rng2 = np.random.default_rng(99999)
            gf_pts, gf_energy = _gradient_flow(
                best_pts, rng2,
                n_steps=50, lr_start=0.001, lr_end=1e-7,
                deadline=deadline - 0.5,
                use_noise=False
            )
            if gf_energy < best_energy:
                best_pts = gf_pts
                best_energy = gf_energy
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY  # ~138s
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Sobol quasi-random sequences with different seeds
    for s in range(3):
        initializations.append(_sobol_sphere(s))
    
    # Jittered Fibonacci spirals with varying amplitudes
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(3):
        amp = 0.02 + 0.015 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(4):
        initializations.append(_random_sphere(100 + s * 13))
    
    # Golden ratio spiral variant
    i = np.arange(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / N
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    theta = golden_angle * i
    golden_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta), z])
    initializations.append(_project(golden_pts))
    
    # Apply repulsive pre-spread to all initializations
    for i in range(len(initializations)):
        if time.time() > deadline - 5.0:
            break
        initializations[i] = _repulsive_spread(initializations[i], n_steps=30, lr=0.005, deadline=deadline - 5.0)
    
    best_pts = None
    best_energy = float('inf')
    
    total_restarts = len(initializations)
    
    # First pass: try all initializations with optimization
    # Allocate time more evenly, with earlier ones getting slightly more
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        # Allocate time: give earlier restarts slightly more time
        # Total budget for restarts is about 100s, leaving 38s for polish
        restart_budget = min(100.0, remaining - 10.0)
        alloc = restart_budget / (total_restarts - idx)
        alloc = max(3.0, min(12.0, alloc))
        
        if remaining - alloc < 10.0:
            break
        
        restart_deadline = time.time() + alloc
        
        # Optimize from this start
        optimized, e_precise = _optimize_from_start(init_pts, restart_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Final high-precision polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        best_pts, best_energy = _final_polish(best_pts, polish_deadline)
    
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
