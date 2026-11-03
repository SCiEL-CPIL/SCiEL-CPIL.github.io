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
_SAFETY = 10.0
_START = time.time()

# Precompute upper triangle indices for N=100
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient using vectorized operations.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Returns (energy, gradient) where gradient has shape (N, 3).
    The gradient is projected onto the tangent space of the sphere.
    """
    n = pts.shape[0]
    # Compute pairwise squared distances
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
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1.0)
    # Extract upper triangle
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


def _sobol_sphere(n, seed=0):
    """Generate n points on the unit sphere using a Sobol quasi-random sequence."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    # Use more samples than needed for better coverage, then take first n
    m = 1 << (n.bit_length())
    samples = sampler.random(m)
    pts = samples[:n]
    # Ensure we don't get near-zero vectors
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    return pts


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)


def _gradient_flow(pts, deadline, n_steps=200):
    """
    Gradient flow with adaptive step size and annealing to escape local minima.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _fast_energy_only(pts)
    current_energy = best_energy
    
    # Adaptive step size with annealing
    base_lr = 0.01
    min_lr = 1e-5
    
    for step in range(n_steps):
        if time.time() > deadline - 2.0:
            break
        
        # Annealing: decrease learning rate over time
        progress = step / n_steps
        lr = base_lr * (1.0 - progress) + min_lr * progress
        
        e, g = _fast_energy_and_grad(pts)
        
        # Take gradient step
        new_pts = pts - lr * g
        new_pts = _project(new_pts)
        
        new_e = _fast_energy_only(new_pts)
        
        # Accept if energy decreased, or with probability based on "temperature"
        if new_e < current_energy:
            pts = new_pts
            current_energy = new_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = new_pts.copy()
        elif step < n_steps * 0.5:
            # Early phase: accept worse solutions with some probability
            delta_e = new_e - current_energy
            accept_prob = math.exp(-delta_e / (0.01 * (1.0 - progress)))
            if np.random.random() < accept_prob:
                pts = new_pts
                current_energy = new_e
            else:
                # Retry with smaller step
                new_pts = pts - lr * 0.1 * g
                new_pts = _project(new_pts)
                new_e = _fast_energy_only(new_pts)
                if new_e < current_energy:
                    pts = new_pts
                    current_energy = new_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pts = new_pts.copy()
    
    return best_pts, best_energy


def _optimize_from_start(pts, deadline):
    """
    Run L-BFGS-B optimization from a starting point, respecting deadline.
    Includes a gradient flow phase to escape local minima.
    """
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
    
    # Phase 1: Coarse optimization with L-BFGS-B
    if time.time() < deadline - 5.0:
        try:
            x0 = pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'maxfun': 500,
                    'ftol': 1e-8,
                    'gtol': 1e-4,
                    'maxls': 20,
                }
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: Gradient flow with annealing to explore and escape local minima
    if time.time() < deadline - 3.0:
        try:
            gf_pts, gf_energy = _gradient_flow(best_pts, deadline, n_steps=100)
            # Refine the gradient flow result with L-BFGS
            if time.time() < deadline - 1.0:
                result = minimize(
                    objective_and_grad,
                    gf_pts.reshape(-1),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 200,
                        'maxfun': 1000,
                        'ftol': 1e-10,
                        'gtol': 1e-6,
                        'maxls': 30,
                    }
                )
                new_pts = _project(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_pts = new_pts
                    best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: Fine refinement
    if time.time() < deadline - 0.5:
        try:
            x0 = best_pts.reshape(-1)
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 300,
                    'maxfun': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 40,
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
    deadline = start_time + 115.0  # 35s safety margin for final polish
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Sobol quasi-random sequences with different seeds
    for s in range(6):
        initializations.append(_sobol_sphere(N, seed=s))
    
    # Jittered Fibonacci spirals with varying amplitudes
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(6):
        amp = 0.02 + 0.01 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(100 + s * 13))
    
    best_pts = None
    best_energy = float('inf')
    
    total_restarts = len(initializations)
    # Allocate time per restart dynamically
    time_per_restart = 90.0 / total_restarts  # ~3-4s per restart
    
    # First pass: try all initializations with optimization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Allocate decreasing time to later restarts
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        # Give earlier restarts more time
        alloc = min(time_per_restart, remaining - 5.0)
        if alloc < 2.0:
            break
        
        restart_deadline = time.time() + alloc
        
        # Optimize from this start
        optimized, e_precise = _optimize_from_start(init_pts, restart_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Final high-precision polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
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
        for polish_pass in range(5):
            if time.time() < deadline - 2.0:
                try:
                    x0 = best_pts.reshape(-1)
                    maxiter = 200 + 100 * polish_pass
                    ftol = 1e-12 - polish_pass * 1e-13
                    if ftol < 1e-15:
                        ftol = 1e-15
                    gtol = 1e-8 - polish_pass * 2e-9
                    if gtol < 1e-11:
                        gtol = 1e-11
                    
                    result = minimize(
                        objective_and_grad,
                        x0,
                        method='L-BFGS-B',
                        jac=True,
                        options={
                            'maxiter': maxiter,
                            'maxfun': 2000 + 500 * polish_pass,
                            'ftol': ftol,
                            'gtol': gtol,
                            'maxls': 40 + 10 * polish_pass,
                        }
                    )
                    new_pts = _project(result.x.reshape(n, 3))
                    e_precise = _precise_energy(new_pts)
                    if e_precise < best_energy:
                        best_energy = e_precise
                        best_pts = new_pts
                except Exception:
                    pass
            else:
                break
    
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
