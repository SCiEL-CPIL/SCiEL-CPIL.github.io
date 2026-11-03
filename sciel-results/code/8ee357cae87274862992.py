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
    
    Parameters:
    pts: (N, 3) array of points.
    
    Returns:
    energy: float
    grad: (N, 3) array, the Riemannian gradient projected to tangent space.
    """
    n = pts.shape[0]
    # Compute pairwise squared distances
    # diff[i, j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient calculation
    # d/dx_i [ -log(||x_i - x_j||) ] = -(x_i - x_j) / ||x_i - x_j||^2
    # Sum over j != i: grad_ambient_i = - sum_{j!=i} (x_i - x_j) / dist_ij^2
    #                   = sum_{j!=i} (x_j - x_i) / dist_ij^2
    
    # W_ij = 1 / dist_ij^2 for i != j, 0 otherwise
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_ambient_i = sum_j W_ij * (x_j - x_i) = sum_j W_ij * x_j - x_i * sum_j W_ij
    # sum_j W_ij * x_j is (W @ pts)_i
    # sum_j W_ij is W.sum(axis=1)
    
    grad_ambient = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    # Project to tangent space: grad_tangent = grad_ambient - (grad_ambient . x_i) * x_i
    # Since x_i is unit vector, x_i . x_i = 1
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

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return pts / norms

def _generate_sobol(n, seed=42):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    # Use enough samples to pick n good ones, or just take n
    # Sobol sequences are low-discrepancy, good for uniform distribution
    # We generate in [0,1)^3 and map to sphere
    # A better mapping: use spherical coordinates derived from Sobol
    # Or just normalize the Sobol points shifted to center
    samples = sampler.random(n)
    # Shift to center of cube [-0.5, 0.5]
    samples = samples - 0.5
    # Normalize
    return _normalize(samples)

def _generate_random(n, rng):
    pts = rng.standard_normal((n, 3))
    return _normalize(pts)

def _generate_fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

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
                'maxiter': 1000,
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-6,
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
                'maxiter': 2000,
                'maxfun': 10000,
                'ftol': 1e-14,
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
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 145s
    deadline = start_time + 145.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci Spiral (often good for spherical distributions)
    initializations.append(_generate_fibonacci(N))
    
    # 2. Sobol Sequence (Quasi-random, very uniform)
    initializations.append(_generate_sobol(N, seed=1))
    initializations.append(_generate_sobol(N, seed=2))
    
    # 3. Random Points
    for _ in range(3):
        initializations.append(_generate_random(N, rng))
        
    # 4. Perturbed Fibonacci
    base = _generate_fibonacci(N)
    for _ in range(2):
        perturbed = base + rng.standard_normal((N, 3)) * 0.05
        initializations.append(_normalize(perturbed))

    best_pts = None
    best_energy = float('inf')
    
    # Sort initializations by quick energy to prioritize good starts
    # Note: _fast_energy is approximate but fast
    scored_inits = []
    for init in initializations:
        e_quick = _fast_energy(init)
        scored_inits.append((e_quick, init))
    scored_inits.sort(key=lambda x: x[0])
    
    for e_quick, init_pts in scored_inits:
        if time.time() > deadline:
            break
            
        optimized_pts, e_precise = _optimize_single(init_pts, deadline)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized_pts

    if best_pts is None:
        best_pts = _generate_fibonacci(N)
        
    # Final normalization
    best_pts = _normalize(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _generate_fibonacci(N)
        best_pts = _normalize(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
