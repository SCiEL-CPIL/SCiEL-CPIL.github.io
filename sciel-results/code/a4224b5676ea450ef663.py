"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _log_energy_flat(x):
    """Compute logarithmic energy from flat vector."""
    pts = x.reshape(-1, 3)
    # Normalize to ensure we are on sphere (L-BFGS-B might drift slightly)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_upper, j_upper = np.triu_indices(N, k=1)
    d = dists[i_upper, j_upper]
    d = np.maximum(d, 1e-12)
    
    terms = np.log(1.0 / d)
    return math.fsum(terms.tolist())

def _gradient_flat(x):
    """Compute gradient from flat vector."""
    pts = x.reshape(-1, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    dists_safe = np.where(dists < 1e-12, 1e-12, dists)
    
    W = np.zeros((N, N))
    mask = ~np.eye(N, dtype=bool)
    W[mask] = 1.0 / (dists_safe[mask] ** 2)
    
    sum_W = W.sum(axis=1, keepdims=True)
    grad = W @ pts - sum_W * pts
    
    # Project to tangent space
    dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot_prod * pts
    
    return grad.flatten()

def _optimize_start(x0):
    """Run L-BFGS-B optimization from a starting point."""
    res = minimize(
        _log_energy_flat,
        x0,
        jac=_gradient_flat,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    return res.x, res.fun

def construct():
    start_time = time.time()
    max_time = 115.0
    
    # Generate multiple starting points
    starts = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts1 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts1, axis=1, keepdims=True)
    pts1 = pts1 / norms
    starts.append(pts1.flatten())
    
    # 2. Random Uniform
    np.random.seed(42)
    pts2 = np.random.randn(N, 3)
    norms = np.linalg.norm(pts2, axis=1, keepdims=True)
    pts2 = pts2 / norms
    starts.append(pts2.flatten())
    
    # 3. Halton Sequence (Low discrepancy)
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    pts3 = np.zeros((N, 3))
    for i in range(N):
        pts3[i, 0] = 2 * halton(i+1, 2) - 1
        pts3[i, 1] = 2 * halton(i+1, 3) - 1
        pts3[i, 2] = 2 * halton(i+1, 5) - 1
        # Normalize to sphere
        norm = np.linalg.norm(pts3[i])
        pts3[i] /= norm
    starts.append(pts3.flatten())

    best_x = None
    best_energy = float('inf')
    
    for x0 in starts:
        if time.time() - start_time > max_time:
            break
            
        try:
            x_opt, energy = _optimize_start(x0)
            if energy < best_energy:
                best_energy = energy
                best_x = x_opt
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to first start
        best_x = starts[0]
        
    # Reshape and normalize
    pts = best_x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    return pts
# EVOLVE-BLOCK-END
