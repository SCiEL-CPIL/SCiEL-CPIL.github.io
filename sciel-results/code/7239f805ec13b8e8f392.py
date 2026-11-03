"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 150.0          # hard kill from evaluator
_SAFETY = 15.0             # safety margin -> work until ~135s
_START = time.time()

# Precompute upper triangle indices for N=100 for speed
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Returns (energy, gradient) where gradient has shape (N, 3).
    The gradient is the Riemannian gradient on the sphere (tangent space).
    """
    n = pts.shape[0]
    # Efficient distance calculation using dot products
    # |x_i - x_j|^2 = 2 - 2 * x_i . x_j for unit vectors
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    
    # Avoid numerical issues on diagonal and very small distances
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    # Energy: -sum_{i<j} log(sqrt(dist_sq)) = -0.5 * sum_{i<j} log(dist_sq)
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    # Gradient computation
    # Euclidean gradient of -0.5 log(dist_sq) wrt x_i:
    # d/dx_i (-0.5 log(2 - 2 x_i.x_j)) = -0.5 * (1/(2-2x_i.x_j)) * (-2 x_j)
    # = x_j / (2 - 2 x_i.x_j) = x_j / dist_sq_ij
    # Sum over j != i: grad_euclid_i = sum_{j!=i} x_j / dist_sq_ij
    # This is (W @ pts)_i where W_ij = 1/dist_sq_ij for i!=j, 0 otherwise.
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_euclid_i = sum_j W_ij x_j
    grad_euclid = W @ pts
    
    # Project gradient onto tangent space: grad_tangent = grad_euclid - (grad_euclid . x_i) * x_i
    dot = np.sum(grad_euclid * pts, axis=1, keepdims=True)
    grad = grad_euclid - dot * pts
    
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


def _sobol_sphere(seed):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(N)
    samples = samples - 0.5
    return _project(samples)


def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """Gradient flow with adaptive learning rate and annealing.
    Helps escape shallow local minima before L-BFGS takes over.
    """
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy_only(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        # Exponential decay of learning rate
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _fast_energy_and_grad(current)
        
        # Adaptive step size to prevent divergence
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-12:
            lr = min(lr, 0.5 / g_norm)
        
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
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: Coarse optimization
    if time.time() < deadline - 5.0:
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
    
    # Phase 2: Medium refinement
    if time.time() < deadline - 5.0:
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
    
    # Phase 3: Fine refinement
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


def _final_polish(pts, deadline):
    """Final ultra-precise polish with tiny perturbations and multiple restarts.
    This helps escape very shallow local minima that L-BFGS might stop at due to tolerance.
    """
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
    # Try a range of small perturbations
    perturbations = [0.0, 5e-4, 3e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    
    for scale in perturbations:
        if time.time() > deadline - 1.0:
            break
        
        # Try a few restarts for non-zero scales to find better directions
        n_restarts = 3 if scale > 0 else 1
        
        for restart in range(n_restarts):
            if time.time() > deadline - 1.0:
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
                    options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
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
    deadline = start_time + 130.0  # 20s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals
    initializations.append(_fibonacci_spiral(N, offset=0.5))
    initializations.append(_fibonacci_spiral(N, offset=0.3))
    initializations.append(_fibonacci_spiral(N, offset=0.7))
    
    # Sobol sequences (better space filling)
    initializations.append(_sobol_sphere(1))
    initializations.append(_sobol_sphere(2))
    
    # Random points
    for s in range(4):
        initializations.append(_random_sphere(100 + s))
    
    # Jittered Fibonacci
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(4):
        jitter_scale = 0.05 + 0.02 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    best_pts = None
    best_energy = float('inf')
    
    # Process each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # 1. Gradient Flow Phase (short, to escape initial bad spots)
        # Use less time for later initializations if running out of time
        gf_steps = 200 if init_idx < 5 else 100
        gf_deadline = deadline - 10.0
        gf_pts, gf_energy = _gradient_flow(init_pts, rng, gf_steps, 0.05, 0.001, gf_deadline)
        
        # 2. L-BFGS Optimization
        opt_pts, opt_energy = _optimize_from_start(gf_pts, deadline)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts
    
    # Final Polish on the global best
    if time.time() < deadline - 2.0 and best_pts is not None:
        final_pts, final_energy = _final_polish(best_pts, deadline)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    # Ensure validity
    if not np.all(np.isfinite(best_pts)) or np.any(np.linalg.norm(best_pts, axis=1) < 1e-12):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
        
    # Check min distance
    d2 = np.einsum('ij,ij->i', best_pts, best_pts)[:, None] + np.einsum('ij,ij->i', best_pts, best_pts)[None, :] \
         - 2.0 * (best_pts @ best_pts.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        # If points are too close, fall back to a safer initialization
        best_pts = _sobol_sphere(42)
        best_pts = _project(best_pts)
        
    return best_pts
