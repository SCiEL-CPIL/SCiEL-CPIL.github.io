"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _energy_and_grad(pts):
    """
    Compute the logarithmic energy and its tangent-space gradient for points on the sphere.
    Energy E = sum_{i<j} log(1 / ||x_i - x_j||) = - sum_{i<j} log(||x_i - x_j||)
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # W_ij = 1 / ||x_i - x_j||^2 for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Ambient gradient of E w.r.t. x_i: sum_j W_ij (x_i - x_j) = W @ x_i - sum_j W_ij x_j
    grad_ambient = W @ pts - W.sum(axis=1, keepdims=True) * pts
    # Project to tangent space
    dot = np.sum(grad_ambient * pts, axis=1, keepdims=True)
    grad_tangent = grad_ambient - dot * pts
    
    return energy, grad_tangent

def _fast_energy(pts):
    """Compute energy quickly for initialization sorting."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
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

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_sobol(n):
    """Generate points using Sobol sequence projected to sphere."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=123)
    samples = sampler.random(n)
    # Map [0,1]^3 to sphere
    # Use a simple mapping: treat as direction vectors
    pts = 2.0 * samples - 1.0
    pts = _normalize(pts)
    return pts

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_perturbed_fibonacci(n, rng, scale):
    base = _generate_fibonacci(n)
    perturbed = base + rng.standard_normal((n, 3)) * scale
    return _normalize(perturbed)

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
        
        # Linearly anneal learning rate
        progress = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _energy_and_grad(current)
        
        # Add small noise for exploration (annealed)
        noise_scale = 0.005 * (1.0 - progress)
        if noise_scale > 1e-6:
            noise = rng.standard_normal((n, 3)) * noise_scale
        else:
            noise = 0.0
        
        # Update in ambient space, then project back to sphere
        current = current - lr * g + noise
        current = _normalize(current)
        
        e_new = _fast_energy(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy

def _optimize_single(pts, deadline):
    """Run L-BFGS-B optimization from a given start with multiple phases."""
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
                'maxiter': 3000,
                'maxfun': 15000,
                'ftol': 1e-14,
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
                'maxiter': 10000,
                'maxfun': 50000,
                'ftol': 1e-16,
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
    
    # Try several tiny perturbations and polish each
    rng = np.random.default_rng(99999)
    perturbations = [0.0, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    
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
                    'maxiter': 20000,
                    'maxfun': 100000,
                    'ftol': 1e-18,
                    'gtol': 1e-14,
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

def _pairwise_swap_improvement(pts, deadline):
    """
    Try local pairwise swaps of nearby points to escape shallow local minima.
    This is a heuristic that can sometimes find better configurations.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Find pairs that are close together (potential for improvement)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Sort pairs by distance
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_sq[iu]
    pair_indices = np.argsort(pair_dists)
    
    # Try swapping the closest pairs
    for idx in pair_indices[:50]:
        if time.time() > deadline:
            break
        
        i, j = iu[0][idx], iu[1][idx]
        if i == j:
            continue
        
        # Create a perturbed version where points i and j are moved slightly
        # toward each other's "ideal" positions
        trial = best_pts.copy()
        
        # Move point i slightly away from j and vice versa
        direction_ij = (pts[i] - pts[j])
        direction_ij = direction_ij / (np.linalg.norm(direction_ij) + 1e-15)
        
        # Small displacement
        delta = 0.01 * direction_ij
        trial[i] = _normalize(pts[i] + delta)[0]
        trial[j] = _normalize(pts[j] - delta)[0]
        
        e_trial = _fast_energy(trial)
        if e_trial < best_energy - 1e-10:
            # Polish this configuration
            polished, e_polished = _optimize_single(trial, deadline)
            if e_polished < best_energy:
                best_pts = polished
                best_energy = e_polished
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral
    initializations.append(_generate_fibonacci(N))
    
    # 2. Sobol sequence
    initializations.append(_generate_sobol(N))
    
    # 3. Random Points
    for i in range(15):
        initializations.append(_generate_random(N, rng))
        
    # 4. Perturbed Fibonacci
    for scale in [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]:
        initializations.append(_generate_perturbed_fibonacci(N, rng, scale))
    
    # 5. Perturbed Sobol
    sobol_base = _generate_sobol(N)
    for scale in [0.01, 0.05, 0.1]:
        perturbed = sobol_base + rng.standard_normal((N, 3)) * scale
        initializations.append(_normalize(perturbed))
    
    # 6. Additional structured: golden ratio with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4]:
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
    
    # Phase A: Gradient flow exploration on top initializations
    flow_deadline = start_time + 35.0
    num_flow = min(10, len(scored_inits))
    for e_quick, init_pts in scored_inits[:num_flow]:
        if time.time() > flow_deadline:
            break
        
        flowed_pts, e_precise = _gradient_flow(init_pts, rng, 800, 0.03, 0.0005, flow_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = flowed_pts
    
    # Phase B: L-BFGS optimization
    if best_pts is not None:
        scored_inits.insert(0, (best_energy, best_pts))
    
    lbfgs_deadline = deadline - 25.0  # Reserve 25s for polish
    for e_quick, init_pts in scored_inits:
        if time.time() > lbfgs_deadline:
            break
            
        optimized_pts, e_precise = _optimize_single(init_pts, lbfgs_deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _generate_fibonacci(N)
        best_energy = _precise_energy(best_pts)
    
    # Phase C: Pairwise swap improvement
    swap_deadline = deadline - 15.0
    if time.time() < swap_deadline:
        best_pts, best_energy = _pairwise_swap_improvement(best_pts, swap_deadline)
    
    # Phase D: Final ultra-precise polish
    polish_deadline = deadline - 2.0  # 2s safety margin
    best_pts, best_energy = _final_polish(best_pts, polish_deadline)
    
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
