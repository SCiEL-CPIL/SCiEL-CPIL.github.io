"""
Logarithmic energy minimization for N points on the unit sphere.
Strategy: Multi-start L-BFGS-B optimization with diverse initializations.
L-BFGS-B is used because it approximates the Hessian, allowing for faster 
convergence and better navigation of the non-convex energy landscape compared 
to first-order methods like gradient descent.
"""
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
    """Project points onto the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """
    Compute the logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)
    Uses matrix operations for speed.
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    # Distance squared matrix
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    # Ensure numerical stability
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    # Sum over upper triangle
    iu = np.triu_indices(n, k=1)
    # Energy is -sum(log(dist))
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """
    Compute energy and gradient.
    Gradient is projected onto the tangent space of the sphere at each point.
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Weight matrix W where W_ij = 1 / dist_ij^2
    # The force between i and j is proportional to (x_i - x_j) / dist_ij^2
    # The gradient of -log(dist_ij) with respect to x_i is (x_j - x_i) / dist_ij^2
    # So grad_i = sum_{j != i} (x_j - x_i) / dist_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W_ij x_j - x_i sum_j W_ij
    row_sums = W.sum(axis=1, keepdims=True)
    grad = (W @ pts) - row_sums * pts
    
    # Project gradient onto tangent space: grad_tan = grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """
    Compute energy using math.fsum for high precision, as required by the evaluator.
    """
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5, jitter=0.0):
    """
    Generate points using a Fibonacci spiral.
    jitter adds random noise to break symmetry and explore nearby minima.
    """
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    if jitter > 0:
        rng = np.random.default_rng(int(offset * 1000))
        pts = pts + jitter * rng.standard_normal((n, 3))
        pts = _project(pts)
        
    return pts


def _random_sphere(rng):
    """Generate random points on the sphere."""
    v = rng.standard_normal((N, 3))
    return _project(v)


def _objective_and_grad(x):
    """
    Objective function for scipy.optimize.minimize.
    Input x is a flat vector of length N*3.
    Returns (energy, gradient).
    """
    pts = x.reshape(N, 3)
    # Project to sphere to maintain constraint during optimization
    pts = _project(pts)
    e, g = _energy_grad(pts)
    return e, g.flatten()


def _optimize_l_bfgsb(pts, deadline, max_iter=5000, tol=1e-12):
    """
    Run L-BFGS-B optimization on a set of points.
    """
    x0 = pts.flatten()
    
    # L-BFGS-B is a first-order method that uses a limited-memory approximation of the Hessian.
    # It is generally very effective for this problem.
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': tol,
                'gtol': 1e-8, # Gradient tolerance
                'maxfun': max_iter * 3
            }
        )
        
        if result.success or result.fun < _energy(pts):
            best_pts = _project(result.x.reshape(N, 3))
            best_e = _energy(best_pts)
            return best_pts, best_e
    except Exception:
        pass
    
    return pts, _energy(pts)


def _valid(X):
    """Check if the configuration is valid."""
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    # Check minimum distance
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - _SAFETY
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Generate a diverse set of initializations
    initializations = []
    
    # Fibonacci spirals with various offsets and jitter levels
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # No jitter (pure spiral)
        initializations.append(_fibonacci_spiral(N, offset=offset, jitter=0.0))
        # Small jitter
        initializations.append(_fibonacci_spiral(N, offset=offset, jitter=0.05))
        # Larger jitter
        initializations.append(_fibonacci_spiral(N, offset=offset, jitter=0.2))
        
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(10):
        initializations.append(_random_sphere(rng))
        
    # "Grid" like initializations (approximate)
    # Try a few different structured patterns
    for theta_step in [1.0, 1.1, 0.9]:
        pts = []
        for i in range(N):
            z = 1.0 - 2.0 * i / N
            theta = i * theta_step * math.pi / 5.0
            r = np.sqrt(np.clip(1.0 - z*z, 0, 1.0))
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
        initializations.append(_project(np.array(pts)))
        
    # 2. Quick evaluation and sorting
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 10.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # 3. Optimize the top candidates
    # We have ~30 initializations. We will optimize the top 5-10.
    num_to_optimize = 8
    
    for e_quick, init_pts in scored_inits[:num_to_optimize]:
        if time.time() > deadline - 5.0:
            break
            
        # Run L-BFGS-B
        opt_pts, opt_e = _optimize_l_bfgsb(init_pts, deadline, max_iter=3000, tol=1e-12)
        
        if opt_e < best_energy:
            best_energy = opt_e
            best_pts = opt_pts
            
    # 4. Final polish on the global best
    if best_pts is not None and time.time() < deadline - 2.0:
        # Run L-BFGS-B again with higher precision / more iterations on the best result
        final_pts, final_e = _optimize_l_bfgsb(best_pts, deadline, max_iter=5000, tol=1e-14)
        if final_e < best_energy:
            best_energy = final_e
            best_pts = final_pts
            
    # 5. Fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
        
    # Final normalization
    best_pts = _project(best_pts)
    
    # Validation
    if not _valid(best_pts):
        # If invalid, fall back to a known good structure
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
        
    return best_pts
