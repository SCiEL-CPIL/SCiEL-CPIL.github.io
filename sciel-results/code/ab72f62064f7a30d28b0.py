"""Logarithmic energy minimization for N points on the unit sphere.
Uses L-BFGS-B style optimization with careful initialization and multiple restarts."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 10.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


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
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of -sum log(d_ij) with respect to points
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
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


def _objective_and_grad(x):
    """Objective function for scipy optimizer: takes flat vector, returns (energy, grad)."""
    pts = x.reshape(N, 3)
    pts = _project(pts)
    e, g = _energy_grad(pts)
    return e, g.flatten()


def _optimize_scipy(pts, deadline, max_iter=10000, tol=1e-12):
    """Use scipy L-BFGS-B to optimize points on sphere."""
    x0 = pts.flatten()
    
    def time_check():
        return time.time() > deadline
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxfun': max_iter * 3}
        )
        
        if result.success or result.fun < _energy(pts):
            best_pts = _project(result.x.reshape(N, 3))
            best_e = _energy(best_pts)
            return best_pts, best_e
    except Exception:
        pass
    
    return pts, _energy(pts)


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-12):
    """Adaptive gradient descent with line search."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    n = N
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        # Adaptive step size
        step = lr / max(g_norm, 1e-10)
        new_pts = _project(current - step * g)  # negative gradient
        
        new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-14:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            lr = min(lr * 1.1, 0.1)  # increase step
        elif new_e < best_e:
            current = new_pts
            lr = lr * 0.9
        else:
            lr = lr * 0.5
            if lr < lr_min:
                break
    
    return best_pts, best_e


def _newton_like(pts, deadline, max_iter=2000, tol=1e-12):
    """Momentum-based gradient descent with adaptive learning rate."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    velocity = np.zeros_like(current)
    beta = 0.9
    lr = 0.05
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        velocity = beta * velocity - lr * g / max(g_norm, 1e-10)
        new_pts = _project(current + velocity)
        
        new_e = _energy(new_pts)
        
        if new_e < best_e:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts
            lr = min(lr * 1.05, 0.5)
        else:
            current = best_pts.copy()
            velocity *= 0.5
            lr *= 0.7
    
    return best_pts, best_e


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
    deadline = start_time + 115.0  # 5s safety margin
    
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
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: quick evaluation of all initializations
    for init_pts in initializations:
        if time.time() > deadline - 20.0:
            break
        
        e_quick = _energy(init_pts)
        if e_quick < best_energy:
            best_energy = e_quick
            best_pts = init_pts.copy()
    
    # Second pass: optimize the top candidates
    if best_pts is not None and time.time() < deadline - 15.0:
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 6 candidates with scipy L-BFGS-B
        top_candidates = scored_inits[:6]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 10.0:
                break
            
            # Use scipy L-BFGS-B for faster convergence
            sp_pts, sp_e = _optimize_scipy(init_pts, deadline, max_iter=5000)
            if sp_e < best_energy:
                best_energy = sp_e
                best_pts = sp_pts
    
    # Final polish on the global best with multiple methods
    if best_pts is not None and time.time() < deadline - 5.0:
        # Gradient descent polish
        refined, ref_e = _gradient_descent(best_pts, deadline, max_iter=5000, lr_init=0.005, lr_min=1e-14)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined
        
        # Momentum-based polish
        if time.time() < deadline - 3.0:
            refined, ref_e = _newton_like(best_pts, deadline, max_iter=3000, tol=1e-14)
            if ref_e < best_energy:
                best_energy = ref_e
                best_pts = refined
        
        # One more scipy pass if time allows
        if time.time() < deadline - 2.0:
            sp_pts, sp_e = _optimize_scipy(best_pts, deadline, max_iter=3000)
            if sp_e < best_energy:
                best_energy = sp_e
                best_pts = sp_pts
    
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
