"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    Returns (energy, gradient) where gradient has shape (N, 3).
    """
    n = pts.shape[0]
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j!=i} (x_i - x_j) / dist_ij^2
    # = sum_{j!=i} (x_j - x_i) / dist_ij^2
    # Note: dist^2 = sum of diff^2
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    # Weight matrix: w_ij = 1/dist_ij^2 for i != j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_j - x_i) = (W @ pts)_i - W[i,:].sum() * x_i
    # Actually: sum_j W_ij * (x_j - x_i) = sum_j W_ij * x_j - x_i * sum_j W_ij
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    # Project gradient onto tangent space: grad_tangent = grad - (grad . x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i+1, n):
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


def _random_points(n, rng):
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts, deadline, rng):
    """Run L-BFGS-B optimization from a starting point."""
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
        # Map gradient back to flat parameterization
        # The gradient in the flat space needs to account for the normalization
        # Simplification: just use the projected gradient on the sphere
        g_flat = g.reshape(-1)
        return e, g_flat
    
    # Use L-BFGS-B
    x0 = pts.reshape(-1)
    try:
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 500,
                'maxfun': 2000,
                'ftol': 1e-12,
                'gtol': 1e-8,
            }
        )
        if result.success or result.fun < best_energy:
            new_pts = result.x.reshape(n, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.clip(norms, 1e-12, None)
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline:
        return best_pts, best_energy
    
    # Try a second phase with tighter tolerances
    try:
        x0 = best_pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 1000,
                'maxfun': 5000,
                'ftol': 1e-15,
                'gtol': 1e-10,
            }
        )
        if result.fun < best_energy:
            new_pts = result.x.reshape(n, 3)
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_pts = new_pts / np.clip(norms, 1e-12, None)
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
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # Fibonacci with different offsets
    initializations.append(_fibonacci_spiral(N, offset=0.3))
    initializations.append(_fibonacci_spiral(N, offset=0.7))
    
    # Random points
    for _ in range(5):
        initializations.append(_random_points(N, rng))
    
    # Slightly perturbed Fibonacci
    base = _fibonacci_spiral(N)
    for _ in range(3):
        perturbed = base + rng.standard_normal((N, 3)) * 0.1
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        initializations.append(perturbed / norms)
    
    best_pts = None
    best_energy = float('inf')
    
    for init_pts in initializations:
        if time.time() > deadline:
            break
        
        # Quick energy check
        e_quick = _fast_energy_only(init_pts)
        if e_quick < best_energy:
            optimized, e_precise = _optimize_from_start(init_pts, deadline, rng)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized
        
        # Even if not better than current best, still try optimizing
        # since different local minima may be found
        if time.time() > deadline:
            break
        optimized, e_precise = _optimize_from_start(init_pts, deadline, rng)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.clip(norms, 1e-12, None)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
