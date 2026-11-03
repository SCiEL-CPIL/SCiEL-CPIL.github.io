"""
Logarithmic energy minimization for N points on the unit sphere.
Combines diverse initializations (Fibonacci, Random, Sobol) with L-BFGS-B
optimization and a final high-precision gradient descent polish using
Armijo line search.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100

def _project(X):
    """Project points onto the unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _energy(pts):
    """
    Compute the logarithmic energy: sum_{i<j} log(1/d_ij).
    Uses vectorized operations for speed.
    """
    n = pts.shape[0]
    # Compute squared distances
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    # Ensure diagonal is 1 (distance to self is 0, but we mask it out)
    # Actually, we only sum over i < j, so diagonal doesn't matter for sum,
    # but we need to avoid log(0) if we were to use the whole matrix.
    # We will extract upper triangle.
    np.fill_diagonal(dist_sq, 1.0) # Avoid log(0) issues if any, though masked
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    # Energy is sum of log(1/d) = -sum(log(d))
    return float(-np.sum(np.log(dist[iu])))

def _energy_grad(pts):
    """
    Compute the energy and its gradient projected onto the tangent space.
    
    E = sum_{i<j} log(1/d_ij)
    dE/dx_i = - sum_{j!=i} (x_i - x_j) / d_ij^2
    
    We compute the raw gradient in R^3 and then project it onto the tangent
    space of the sphere at x_i, which is the subspace orthogonal to x_i.
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Weight matrix W where W_ij = 1/d_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Raw gradient term: sum_{j!=i} (x_i - x_j) W_ij
    # = x_i * sum_j W_ij - sum_j W_ij x_j
    row_sums = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = row_sums * pts - Wx
    
    # The true gradient of E is -raw_grad
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad

def _precise_energy(pts):
    """Compute energy using math.fsum for high accuracy."""
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

def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence mapped from 2D unit square."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    # Map [0,1]^2 to sphere uniformly
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]

def _optimize_lbfgs(pts, max_iter=3000, ftol=1e-14, gtol=1e-12):
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
                'maxfun': max_iter * 20,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts

def _gradient_descent_polish(pts, max_iter=50000, tol=1e-14):
    """
    Custom gradient descent with Armijo line search for final polish.
    This helps extract more precision from the local minimum.
    """
    n = pts.shape[0]
    best_pts = pts.copy()
    best_e = _energy(pts)
    
    # Initial step size
    step = 0.05
    c1 = 1e-4  # Armijo parameter
    
    for it in range(max_iter):
        e, g = _energy_grad(best_pts)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Armijo line search
        # Direction is -g (since g is gradient of E, we want to go opposite)
        # Wait, _energy_grad returns the gradient of E.
        # So descent direction is -g.
        direction = -g
        
        # Try to find a step size that satisfies Armijo condition
        # E(x + alpha * d) <= E(x) + c1 * alpha * grad_E . d
        # grad_E . d = grad_E . (-grad_E) = -||grad_E||^2
        # So condition: E_new <= E_old - c1 * alpha * ||grad_E||^2
        
        accepted = False
        alpha = step
        for _ in range(20):
            candidate = best_pts + alpha * direction
            candidate = _project(candidate)
            cand_e = _energy(candidate)
            
            if cand_e <= best_e - c1 * alpha * (g_norm ** 2):
                best_e = cand_e
                best_pts = candidate
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if accepted:
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.7
            if step < 1e-12:
                break
        
        if time.time() > 145.0: # Internal deadline guard
            break

    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin from 150s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy, good for global coverage)
    for s in range(10):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.01 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points
    for s in range(10):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # Score all initializations quickly
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Optimize top candidates with L-BFGS-B
    # We can afford to optimize more candidates if they are fast
    top_candidates = scored_inits[:15]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 20.0:
            break
        
        optimized = _optimize_lbfgs(init_pts, max_iter=2000, ftol=1e-14, gtol=1e-10)
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
        # Use gradient descent polish with Armijo for high precision
        polished = _gradient_descent_polish(best_pts, max_iter=20000, tol=1e-14)
        e_precise = _precise_energy(polished)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    return best_pts

if __name__ == '__main__':
    pts = construct()
    print(_precise_energy(pts))
