import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
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
    dist = np.sqrt(np.maximum(dist_sq, 1e-16))
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
    # d/dx_i [-log(d_ij)] = -(x_i - x_j) / d_ij^2
    # Sum over j != i: -sum_j (x_i - x_j)/d_ij^2 = -sum_j x_i/d_ij^2 + sum_j x_j/d_ij^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = -sum_j W_ij * x_i + sum_j W_ij * x_j = -W.sum(axis=1)*x_i + W @ x
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


def _lbfgs_penalized(pts, deadline, max_iter=2000, penalty=100.0):
    """
    Optimize using L-BFGS-B with a penalty term for sphere constraint.
    """
    x0 = pts.flatten()
    
    def objective_and_grad(x):
        X = x.reshape(N, 3)
        # Calculate energy and its Euclidean gradient
        sq_norms = np.einsum('ij,ij->i', X, X)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (X @ X.T)
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-16)
        dist = np.sqrt(dist_sq)
        
        iu = np.triu_indices(N, k=1)
        energy = float(-np.sum(np.log(dist[iu])))
        
        W = 1.0 / dist_sq
        np.fill_diagonal(W, 0.0)
        row_sums = W.sum(axis=1, keepdims=True)
        grad_euclidean = -row_sums * X + (W @ X)
        
        # Penalty term: lambda * sum(||x_i||^2 - 1)^2
        norms_sq = sq_norms
        penalty_val = penalty * np.sum((norms_sq - 1.0) ** 2)
        
        # Gradient of penalty: 4 * lambda * (||x_i||^2 - 1) * x_i
        grad_penalty = 4.0 * penalty * (norms_sq - 1.0)[:, np.newaxis] * X
        
        total_grad = grad_euclidean + grad_penalty
        total_energy = energy + penalty_val
        
        return total_energy, total_grad.flatten()

    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, 
                   options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': max_iter*2})
    
    if res.success or res.status == 0:
        best_pts = _project(res.x.reshape(N, 3))
        best_e = _energy(best_pts)
        return best_pts, best_e
    else:
        # Return the best found so far if failed? L-BFGS-B returns the last iterate.
        # We should check if the last iterate is better.
        best_pts = _project(res.x.reshape(N, 3))
        best_e = _energy(best_pts)
        return best_pts, best_e


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
        new_pts = _project(current + step * g * (-1))  # negative gradient
        
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
    """Second-order method using Hessian approximation via finite differences."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    # Simple momentum-based gradient descent with adaptive learning rate
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
    
    # Sobol-like points
    for s in range(5):
        initializations.append(_sobol_like(N, seed=s))
    
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
    
    # Second pass: optimize the top candidates using L-BFGS-B with penalty
    if best_pts is not None and time.time() < deadline - 15.0:
        # Sort initializations by quick energy and take top ones
        scored_inits = []
        for init_pts in initializations:
            e_quick = _energy(init_pts)
            scored_inits.append((e_quick, init_pts))
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize top 10 candidates
        top_candidates = scored_inits[:10]
        for e_quick, init_pts in top_candidates:
            if time.time() > deadline - 10.0:
                break
            
            lbfgs_pts, lbfgs_e = _lbfgs_penalized(init_pts, deadline, max_iter=500, penalty=100.0)
            if lbfgs_e < best_energy:
                best_energy = lbfgs_e
                best_pts = lbfgs_pts
            
            # Also try gradient descent if time permits
            if time.time() < deadline - 5.0:
                gd_pts, gd_e = _gradient_descent(init_pts, deadline, max_iter=2000, lr_init=0.01)
                if gd_e < best_energy:
                    best_energy = gd_e
                    best_pts = gd_pts
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
        refined, ref_e = _lbfgs_penalized(best_pts, deadline, max_iter=1000, penalty=1000.0)
        if ref_e < best_energy:
            best_energy = ref_e
            best_pts = refined
        
        # Final gradient descent polish
        if time.time() < deadline - 2.0:
            refined, ref_e = _gradient_descent(best_pts, deadline, max_iter=5000, lr_init=0.001, lr_min=1e-14)
            if ref_e < best_energy:
                best_energy = ref_e
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
