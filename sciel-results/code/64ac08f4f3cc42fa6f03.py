"""Logarithmic energy minimization for N points on the unit sphere.
Search-based approach with spectral basis initialization and multi-phase optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 148.0  # safety margin from 150s limit
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
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
    
    # Gradient: dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij * (x_i - x_j) = (W @ x)_i - (W * x)_i... 
    # Actually: sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    grad = W.sum(axis=1, keepdims=True) * pts - W @ pts
    # Project onto tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


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


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_like(n, seed=0):
    """Generate well-distributed points using a simple quasi-random approach."""
    rng = np.random.default_rng(seed)
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        t = (i + 0.5) / n
        z = 1.0 - 2.0 * t
        angle = 2.0 * np.pi * i * (1.0 + 1.0 / golden_ratio) + rng.standard_normal() * 0.1
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
        points.append([r * np.cos(angle), r * np.sin(angle), z])
    return _project(np.array(points))


def _spectral_initialization(n_points, l_max=8, rng=None):
    """
    Initialize points using spherical harmonic coefficients.
    We parameterize the density on the sphere as a sum of spherical harmonics
    up to degree l_max, then sample points from this density.
    
    This gives a smooth, low-frequency initial configuration that may escape
    shallow local minima.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # We'll use a different approach: generate points via iterative refinement
    # starting from a spectral basis representation
    
    # Method: Start with a set of points, compute their "spectral representation"
    # (projection onto low-degree spherical harmonics), then reconstruct a smoother
    # configuration and sample from it.
    
    # For now, use a hybrid: Fibonacci spiral with controlled noise at different scales
    base = _fibonacci_spiral(n_points, offset=0.5)
    
    # Add multi-scale noise
    result = base.copy()
    for scale in [0.1, 0.05, 0.02]:
        noise = rng.standard_normal((n_points, 3)) * scale
        result = _project(result + noise)
    
    return result


def _optimize(pts, deadline, max_iter=1000, ftol=1e-15, gtol=1e-12, maxls=50):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': maxls,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _gradient_descent_polish(pts, deadline, lr=0.01, decay=0.999, max_steps=5000):
    """Simple gradient descent with adaptive step size for final polish."""
    n = pts.shape[0]
    current = pts.copy()
    best = current.copy()
    best_e = _energy(current)
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        
        if e < best_e:
            best_e = e
            best = current.copy()
        
        # Adaptive step size
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        
        # Take step
        new_pts = _project(current - lr * g)
        new_e = _energy(new_pts)
        
        if new_e < e:
            current = new_pts
            # Occasionally increase lr if we're making good progress
            if step % 100 == 0 and step > 0:
                lr = min(lr * 1.05, 0.1)
        else:
            lr *= decay
            if lr < 1e-10:
                break
    
    return best


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Sobol-like points
    for s in range(5):
        initializations.append(_sobol_like(N, seed=s))
    
    # Spectral-like initializations
    for s in range(5):
        initializations.append(_spectral_initialization(N, rng=np.random.default_rng(200 + s)))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Second pass: optimize the top candidates
    if best_pts is not None and time.time() < deadline - 20.0:
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 10 candidates
        top_candidates = scored_inits[:10]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 15.0:
                break
            
            # Phase 1: coarse optimization
            optimized = _optimize(init_pts, deadline, max_iter=300, ftol=1e-10, gtol=1e-6, maxls=30)
            
            # Phase 2: finer optimization
            if time.time() < deadline - 10.0:
                optimized = _optimize(optimized, deadline, max_iter=500, ftol=1e-13, gtol=1e-9, maxls=40)
            
            e_precise = _precise_energy(optimized)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = optimized
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 10.0:
        # High-precision L-BFGS-B
        refined = _optimize(best_pts, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
    # Gradient descent polish
    if best_pts is not None and time.time() < deadline - 5.0:
        polished = _gradient_descent_polish(best_pts, deadline, lr=0.005, max_steps=3000)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    # Another refinement pass
    if best_pts is not None and time.time() < deadline - 3.0:
        refined = _optimize(best_pts, deadline, max_iter=1000, ftol=1e-16, gtol=1e-13, maxls=50)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
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
