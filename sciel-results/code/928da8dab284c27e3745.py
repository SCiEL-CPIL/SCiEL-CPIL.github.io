"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

_DEADLINE = 115.0  # seconds; safety margin before 120s kill

def _energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return 1e18
            total += math.log(1.0 / d)
    return total

def _energy_fast(pts):
    """Vectorized energy computation."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return 1e18
    return float(np.sum(np.log(1.0 / d)))

def _energy_grad(pts):
    """Compute energy and gradient on the sphere using tangent space."""
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)  # avoid division by zero
    
    # Energy: sum over i<j of log(1/d_ij)
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    if np.any(d < 1e-9):
        return 1e18, np.zeros_like(pts)
    E = float(np.sum(np.log(1.0 / d)))
    
    # Gradient computation
    # For each pair (i,j), contribution to grad[i] is -(x_i - x_j) / d_ij^2
    # and to grad[j] is (x_i - x_j) / d_ij^2
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = dist[i, j]
            if d_ij < 1e-9:
                continue
            diff_ij = pts[i] - pts[j]
            contrib = diff_ij / (d_ij * d_ij)
            grad[i] += -contrib
            grad[j] += contrib
    
    # Project gradient to tangent space
    grad_tan = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return E, grad_tan

def _project(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _gradient(pts):
    """Compute gradient of energy w.r.t. point positions (before projection)."""
    n = len(pts)
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-9:
                continue
            # d/dx_i of log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|^2
            # = -(x_i - x_j) / d^2
            grad[i] += -diff / (d * d)
    return grad

def _cg_optimize(pts, max_iter=2000, tol=1e-10):
    """Polak-Ribiere conjugate gradient optimization on the sphere."""
    pts = _project(pts.copy())
    E = _energy_fast(pts)
    g = _gradient(pts)
    # Project gradient to tangent space
    g_tan = g - np.sum(g * pts, axis=1, keepdims=True) * pts
    s = -g_tan
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        # Line search
        alpha = 0.01
        for _ in range(30):
            new_pts = _project(pts + alpha * s)
            new_E = _energy_fast(new_pts)
            if new_E < E:
                break
            alpha *= 0.5
        else:
            break
        
        pts = new_pts
        E = new_E
        
        new_g = _gradient(pts)
        new_g_tan = new_g - np.sum(new_g * pts, axis=1, keepdims=True) * pts
        
        if np.linalg.norm(new_g_tan) < tol:
            break
        
        beta = np.dot(new_g_tan, new_g_tan - g_tan) / (np.dot(g_tan, g_tan) + 1e-15)
        beta = max(0, beta)
        s = -new_g_tan + beta * s
        g_tan = new_g_tan
    
    return pts, E

def _lbfgs_optimize(pts, max_iter=3000, tol=1e-8):
    """L-BFGS-B optimization using scipy, with tangent space handling."""
    from scipy.optimize import minimize
    
    pts = _project(pts.copy())
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        E, grad_tan = _energy_grad(x)
        # For L-BFGS, we need the gradient in the flattened parameter space
        # But we're optimizing on the sphere, so we use a retraction approach
        # Here we just return the tangent gradient for now
        return E, grad_tan.flatten()
    
    x0 = pts.flatten()
    
    # Use L-BFGS-B with bounds to keep points near the sphere
    # Actually, we'll use a different approach: optimize in ambient space with penalty
    def penalized_objective(x_flat):
        x = x_flat.reshape(N, 3)
        E, _ = _energy_grad(x)
        # Penalty for being off the sphere
        norms = np.linalg.norm(x, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0)**2)
        return E + penalty
    
    def penalized_grad(x_flat):
        x = x_flat.reshape(N, 3)
        E, grad_tan = _energy_grad(x)
        # Add penalty gradient
        norms = np.linalg.norm(x, axis=1)
        penalty_grad = 2e6 * (norms - 1.0) * x / np.maximum(norms, 1e-15)
        return (grad_tan + penalty_grad).flatten()
    
    result = minimize(penalized_objective, x0, jac=penalized_grad, 
                     method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': tol})
    
    opt_pts = _project(result.x.reshape(N, 3))
    opt_E = _energy_fast(opt_pts)
    
    return opt_pts, opt_E

def _random_start(seed):
    """Generate a random set of points on the sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((N, 3))
    return _project(pts)

def _fibonacci_start():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _project(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _jittered_fibonacci(jitter, seed):
    """Fibonacci start with small random jitter."""
    pts = _fibonacci_start()
    rng = np.random.default_rng(seed)
    pts = pts + rng.standard_normal((N, 3)) * jitter
    return _project(pts)

def construct():
    start_time = time.time()
    
    best_pts = None
    best_E = 1e18
    
    # Try multiple diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(('fib', _fibonacci_start()))
    
    # 2. Random starts
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > _DEADLINE - 10:
            break
        initializations.append((f'rand_{seed}', _random_start(seed)))
    
    # 3. Jittered Fibonacci
    for jitter in [0.01, 0.05, 0.1]:
        if time.time() > _DEADLINE - 10:
            break
        initializations.append((f'jitter_{jitter}', _jittered_fibonacci(jitter, seed=77)))
    
    # Phase 1: CG optimization on all initializations
    for name, init_pts in initializations:
        if time.time() > _DEADLINE - 20:
            break
        
        opt_pts, E = _cg_optimize(init_pts, max_iter=1500)
        
        if E < best_E:
            best_E = E
            best_pts = opt_pts
    
    # Phase 2: L-BFGS-B polish on the best result from Phase 1
    if best_pts is not None and time.time() < _DEADLINE - 15:
        try:
            polished_pts, polished_E = _lbfgs_optimize(best_pts, max_iter=2000, tol=1e-9)
            if polished_E < best_E:
                best_E = polished_E
                best_pts = polished_pts
        except Exception:
            pass
    
    # Phase 3: Additional CG polish on the best result
    if best_pts is not None and time.time() < _DEADLINE - 5:
        opt_pts, E = _cg_optimize(best_pts, max_iter=500, tol=1e-12)
        if E < best_E:
            best_E = E
            best_pts = opt_pts
    
    if best_pts is None:
        # Fallback: return Fibonacci if something went wrong
        return _fibonacci_start()
    
    return _project(best_pts)
# EVOLVE-BLOCK-END
