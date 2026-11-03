"""Logarithmic energy minimization for 100 points on the unit sphere.
Combines low-discrepancy initializations (Sobol, Fibonacci) with tangent-space
L-BFGS-B optimization and a final high-precision polish."""

import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 143.0  # safety margin from 150s limit


def _project(X):
    """Normalize points to unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy_and_grad(pts):
    """Compute log energy and tangent-space gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = -sum log(d_ij)
    # dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx  # sum (x_i - x_j)/d^2
    grad = -raw_grad  # true gradient
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for maximum accuracy."""
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
    """Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project(pts)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence (low discrepancy)."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _tangent_to_sphere(pts, tvec):
    """Map tangent vectors to sphere via exponential map for better accuracy."""
    tv = tvec.reshape(N, 3)
    # Project to tangent space
    dots = np.sum(tv * pts, axis=1)
    tv = tv - dots[:, np.newaxis] * pts
    
    # Use exponential map: x_new = cos(|v|) * x + sin(|v|) * v/|v|
    tv_norm = np.linalg.norm(tv, axis=1, keepdims=True)
    # Handle small norms to avoid division by zero
    safe_norm = np.where(tv_norm < 1e-12, 1.0, tv_norm)
    tv_unit = tv / safe_norm
    
    cos_t = np.cos(tv_norm)
    sin_t_over_norm = np.sin(tv_norm) / safe_norm
    
    new_pts = cos_t * pts + sin_t_over_norm * tv
    return _project(new_pts)


def _lbfgs_optimize(pts0, max_iter=3000, ftol=1e-15, gtol=1e-12):
    """Run L-BFGS-B in tangent space parameterization."""
    n = pts0.shape[0]
    
    def _obj_and_grad(tvec):
        pts = _tangent_to_sphere(pts0, tvec)
        e, g = _energy_and_grad(pts)
        # For small tvec, the chain rule gives grad_tvec ≈ g (already tangent)
        return e, g.reshape(-1)
    
    try:
        res = minimize(_obj_and_grad, np.zeros(3 * n), method='L-BFGS-B',
                       jac=True, options={'maxiter': max_iter, 'ftol': ftol, 
                                          'gtol': gtol, 'maxls': 100, 'maxfun': 20000,
                                          'iprint': -1})
        pts_opt = _tangent_to_sphere(pts0, res.x)
        return pts_opt, res.fun
    except Exception:
        return pts0, np.inf


def _coordinate_descent_polish(pts, max_rounds=50, time_budget=3.0):
    """Final coordinate-descent polish: for each point, optimize its position
    on the sphere given all others fixed, using small tangent-space steps."""
    start = time.time()
    pts = _project(pts.copy())
    n = pts.shape[0]
    
    for rnd in range(max_rounds):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(n):
            # Compute gradient component for point i only
            dists = np.linalg.norm(pts[i] - pts, axis=1)
            dists[i] = 1.0  # avoid self
            dists = np.maximum(dists, 1e-12)
            
            # grad_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
            W = 1.0 / (dists ** 2)
            W[i] = 0.0
            raw_g = W.sum() * pts[i] - (W * pts).sum(axis=0)
            
            # Project to tangent
            dot = np.dot(raw_g, pts[i])
            g_tan = raw_g - dot * pts[i]
            
            # Small step in tangent direction
            g_norm = np.linalg.norm(g_tan)
            if g_norm < 1e-15:
                continue
            
            # Adaptive step size
            step = min(0.05, 0.01 / (g_norm + 1e-10))
            new_pt = pts[i] + step * g_tan
            new_pt = new_pt / np.linalg.norm(new_pt)
            
            # Check if energy decreased
            old_e = _precise_energy(pts)
            pts_old_i = pts[i].copy()
            pts[i] = new_pt
            new_e = _precise_energy(pts)
            
            if new_e < old_e - 1e-12:
                improved = True
            else:
                pts[i] = pts_old_i
        
        if not improved:
            break
    
    return _project(pts), _precise_energy(pts)


def _local_optimize(pts, time_budget=10.0):
    """Local optimization with L-BFGS + coordinate descent polish."""
    start = time.time()
    pts = _project(pts.copy())
    best_pts = pts.copy()
    best_energy = _precise_energy(best_pts)
    
    # Phase 1: Standard optimization
    try:
        pts_opt, e_opt = _lbfgs_optimize(pts, max_iter=3000, ftol=1e-14, gtol=1e-10)
        if e_opt < best_energy:
            best_pts = pts_opt
            best_energy = e_opt
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed > time_budget * 0.6:
        # Still try a quick coordinate descent polish
        if time.time() - start < time_budget * 0.9:
            try:
                cd_pts, cd_e = _coordinate_descent_polish(best_pts, max_rounds=20, time_budget=min(2.0, time_budget - (time.time() - start)))
                if cd_e < best_energy:
                    best_pts = cd_pts
                    best_energy = cd_e
            except Exception:
                pass
        return best_pts, best_energy
    
    # Phase 2: Polish with tighter tolerances
    try:
        pts_opt2, e_opt2 = _lbfgs_optimize(best_pts, max_iter=5000, ftol=1e-16, gtol=1e-14)
        if e_opt2 < best_energy:
            best_pts = pts_opt2
            best_energy = e_opt2
    except Exception:
        pass
    
    # Phase 3: Coordinate descent polish
    remaining = time_budget - (time.time() - start)
    if remaining > 0.5:
        try:
            cd_pts, cd_e = _coordinate_descent_polish(best_pts, max_rounds=50, time_budget=min(3.0, remaining))
            if cd_e < best_energy:
                best_pts = cd_pts
                best_energy = cd_e
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations: Sobol, Fibonacci, random
    configs = []
    
    # Sobol sequences (low discrepancy, 15 seeds)
    for s in range(15):
        configs.append(_sobol_sphere(N, seed=s))
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        configs.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci with various scales
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.02 + 0.015 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        configs.append(_project(jittered))
    
    # Random configurations
    for s in range(10):
        configs.append(_random_sphere(N, np.random.default_rng(100 + s)))
    
    # Jittered Sobol
    for s in range(8):
        sobol_pts = _sobol_sphere(N, seed=s)
        jitter_scale = 0.02 + 0.01 * s
        jittered = sobol_pts + jitter_scale * rng.standard_normal((N, 3))
        configs.append(_project(jittered))
    
    # Score initializations quickly
    scored = []
    for cfg in configs:
        if time.time() > deadline - 25.0:
            break
        e_quick = _precise_energy(cfg)
        scored.append((e_quick, cfg))
    
    scored.sort(key=lambda x: x[0])
    
    # Optimize top candidates with adaptive time budget
    best_pts = None
    best_energy = float('inf')
    top_k = min(30, len(scored))
    
    # Track which initializations have been tried to avoid redundancy
    tried_energies = set()
    
    for i in range(top_k):
        if time.time() > deadline - 15.0:
            break
        
        e_quick, init_pts = scored[i]
        
        # Skip if we've already found a very similar solution
        if best_pts is not None:
            # Quick check: if initial energy is much worse than current best, skip
            if e_quick > best_energy + 5.0 and i > 5:
                continue
        
        # Allocate time: more for earlier (better) candidates
        if i < 3:
            time_budget = 12.0
        elif i < 10:
            time_budget = 8.0
        elif i < 20:
            time_budget = 5.0
        else:
            time_budget = 3.0
        
        pts_opt, e_opt = _local_optimize(init_pts, time_budget=time_budget)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
    
    # Final polish on the global best with generous time
    if best_pts is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 8.0:
            best_pts, best_energy = _local_optimize(best_pts, time_budget=remaining)
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final high-precision coordinate descent polish
    if time.time() < deadline - 1.0:
        try:
            cd_pts, cd_e = _coordinate_descent_polish(best_pts, max_rounds=100, time_budget=min(5.0, deadline - time.time() - 0.5))
            if cd_e < best_energy:
                best_pts = cd_pts
                best_energy = cd_e
        except Exception:
            pass
    
    return best_pts
