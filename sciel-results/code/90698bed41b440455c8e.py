import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|xi-xj|)"""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))

def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
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
    
    # Project onto tangent space: grad - (grad . x) x
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_proj = grad - dot * pts
    
    return energy, grad_proj

def _fibonacci_spiral(n, offset=0.5, jitter=0.0, seed=42):
    """Generate points on sphere using Fibonacci spiral with optional offset and jitter."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        x += jitter * rng.standard_normal((n, 3))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / norms
    return x

def _sobol_sphere(n, seed=42):
    """Generate points on sphere using Sobol sequence for quasi-random uniform coverage."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    samples = sampler.random(m)
    theta = np.arccos(1 - 2 * samples[:, 0])
    phi = 2 * np.pi * samples[:, 1]
    x = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return x[:n]

def _random_sphere(rng):
    """Generate random points on sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _repulsive_pre_opt(pts, deadline, max_iter=200):
    """
    Pre-optimization step using repulsive energy to spread out points.
    This helps avoid local minima where points cluster.
    """
    def repulsive_energy_and_grad(flat):
        if time.time() > deadline:
            return 1e10, np.zeros_like(flat)
        p = _project(flat.reshape(N, 3))
        # Compute pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        
        # Repulsive energy: sum 1/d (stronger repulsion for close points)
        rep_energy = float(np.sum(1.0 / dist[iu]))
        
        # Gradient of sum 1/d wrt x_i
        # d(1/d)/dx_i = -1/d^2 * (x_i - x_j)/d = -(x_i - x_j)/d^3
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = diff / (dist_sq[:, :, np.newaxis] * dist[:, :, np.newaxis])
        
        idx = np.diag_indices(N)
        terms[idx] = 0.0
        grad = -np.sum(terms, axis=1)
        
        # Project onto tangent space
        dot = np.sum(grad * p, axis=1, keepdims=True)
        grad_proj = grad - dot * p
        
        return rep_energy, grad_proj.flatten()
    
    try:
        result = minimize(
            repulsive_energy_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'gtol': 1e-10, 'disp': False, 'ftol': 1e-12}
        )
        if result.fun < 1e9:
            return _project(result.x.reshape(N, 3))
    except Exception:
        pass
    
    return pts

def _optimize(pts, deadline, maxiter=500, ftol=1e-15, gtol=1e-12):
    """Optimize points using L-BFGS-B with time limit."""
    def objective_and_grad(flat):
        if time.time() > deadline:
            return 1e10, np.zeros_like(flat)
        pts_local = _project(flat.reshape(N, 3))
        energy, grad = _energy_grad(pts_local)
        return energy, grad.flatten()
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'disp': False, 'ftol': ftol}
        )
        if result.fun < 1e9:
            return _project(result.x.reshape(N, 3)), float(result.fun)
    except Exception:
        pass
    
    return pts, _energy(pts)

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_x = None
    best_energy = float('inf')
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with different offsets
    for i in range(5):
        initializations.append(_fibonacci_spiral(N, offset=0.1 + 0.15 * i, seed=i))
    
    # Sobol sequences with different seeds
    for i in range(3):
        initializations.append(_sobol_sphere(N, seed=100 + i))
    
    # Random points
    rng = np.random.default_rng(42)
    for i in range(2):
        pts = rng.standard_normal((N, 3))
        initializations.append(_project(pts))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(3):
        jitter_scale = 0.05 + 0.02 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Phase 2: Repulsive pre-optimization for each initialization
    # This spreads out points before the main logarithmic energy optimization
    pre_optimized = []
    for i, pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        pts_pre = _repulsive_pre_opt(pts, deadline, max_iter=100)
        pre_optimized.append(pts_pre)
    
    # Phase 3: Main logarithmic energy optimization
    for i, pts in enumerate(pre_optimized):
        if time.time() > deadline - 15:
            break
        
        time_left = deadline - time.time()
        if time_left < 30:
            maxiter = 300
        else:
            maxiter = 500
            
        x_opt, e_opt = _optimize(pts, deadline, maxiter=maxiter, ftol=1e-15, gtol=1e-12)
        
        # Use precise energy for comparison
        e_precise = _precise_energy(x_opt)
        if e_precise < best_energy:
            best_energy = e_precise
            best_x = x_opt.copy()
    
    # Phase 4: Final high-precision polish on the best solution
    if best_x is not None and time.time() < deadline - 5:
        x_polished, e_polished = _optimize(best_x, deadline, maxiter=1000, ftol=1e-16, gtol=1e-14)
        e_precise = _precise_energy(x_polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_x = x_polished.copy()
    
    # Final check: ensure valid output
    if best_x is None:
        # Fallback to a simple Fibonacci spiral if optimization failed
        best_x = _fibonacci_spiral(N)
        
    best_x = _project(best_x)
    assert best_x.shape == (N, 3)
    assert np.all(np.isfinite(best_x))
    assert np.all(np.linalg.norm(best_x, axis=1) > 1e-9)
    
    return best_x
