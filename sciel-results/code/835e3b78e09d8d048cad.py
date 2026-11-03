"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

# EVOLVE-BLOCK-START
def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    grad_ambient = W @ pts - W.sum(axis=1, keepdims=True) * pts
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Compute energy quickly for initialization sorting."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.1)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))

def _precise_energy(pts):
    """Compute energy using math.fsum for final scoring accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_sobol(n, seed=42):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(n)
    samples = samples - 0.5
    return _normalize(samples)

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """
    Perform gradient flow with annealing to explore configuration space.
    Uses tangent-space gradient descent with adaptive learning rate.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _energy_and_grad(current)
        
        noise_scale = 0.005 * (1.0 - progress)
        if noise_scale > 0:
            noise = rng.standard_normal((n, 3)) * noise_scale
        else:
            noise = np.zeros((n, 3))
        
        current = current - lr * g + noise
        current = _normalize(current)
        
        e_new = _fast_energy(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy

def _optimize_single(pts, deadline):
    """Run L-BFGS-B optimization from a given start."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    x0 = pts.reshape(-1)
    
    # Phase 1: Standard L-BFGS-B
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-12,
                'gtol': 1e-8,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Phase 2: Polish with tighter tolerances
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'maxfun': 20000,
                'ftol': 1e-15,
                'gtol': 1e-10,
            }
        )
        if result.fun < best_energy:
            new_pts = _normalize(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    return best_pts, best_energy

def _final_polish(pts, deadline):
    """
    Final ultra-precise polish pass with very tight tolerances.
    Uses multiple L-BFGS restarts from slightly perturbed versions.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _normalize(p)
        e, g = _energy_and_grad(p)
        return e, g.reshape(-1)
    
    rng = np.random.default_rng(12345)
    perturbations = [0.0] + [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    
    for scale in perturbations:
        if time.time() > deadline:
            break
        
        if scale == 0.0:
            start = best_pts
        else:
            start = _normalize(best_pts + rng.standard_normal((n, 3)) * scale)
        
        try:
            result = minimize(
                objective_and_grad,
                start.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 10000,
                    'maxfun': 50000,
                    'ftol': 1e-18,
                    'gtol': 1e-12,
                }
            )
            if result.fun < best_energy:
                new_pts = _normalize(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_pts = new_pts
                    best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_generate_fibonacci(N))
    
    # 2. Sobol Sequences
    for seed in [1, 2, 3]:
        initializations.append(_generate_sobol(N, seed=seed))
    
    # 3. Random Points
    for _ in range(8):
        initializations.append(_generate_random(N, rng))
        
    # 4. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for scale in [0.01, 0.05, 0.1]:
        perturbed = base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # 5. Structured golden-ratio variations
    for offset in [0.0, 0.1, 0.2]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        initializations.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
    
    # Sort initializations by quick energy to prioritize good starts
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Time budget allocation: ~60% for gradient flow + L-BFGS, ~40% for final polish
    main_deadline = start_time + 85.0
    
    for e_quick, init_pts in scored_inits:
        if time.time() > main_deadline:
            break
        
        # Gradient flow phase to explore and escape local minima
        gf_pts, gf_energy = _gradient_flow(init_pts, rng, n_steps=500, lr_start=0.01, lr_end=0.001, deadline=main_deadline)
        
        # L-BFGS-B optimization
        optimized_pts, e_precise = _optimize_single(gf_pts, main_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts
    
    if best_pts is None:
        best_pts = _generate_fibonacci(N)
        best_energy = _precise_energy(best_pts)
    
    # Final ultra-precise polish
    best_pts, best_energy = _final_polish(best_pts, deadline)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
