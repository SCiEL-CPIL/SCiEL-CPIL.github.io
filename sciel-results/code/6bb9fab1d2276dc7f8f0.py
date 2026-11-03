"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    pts: (N, 3) array of points on the unit sphere
    Returns: (energy, gradient) where gradient is the unconstrained gradient
             (caller must project to tangent space)
    """
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (N, N)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    inv_dist_sq = 1.0 / dist_sq  # (N, N)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sum = inv_dist_sq.sum(axis=1)  # (N,)
    W_pts = inv_dist_sq @ pts  # (N, 3)
    # grad_i = sum_j (x_j - x_i) / dist_ij^2 = sum_j W[i,j]*x_j - x_i * sum_j W[i,j]
    grad = W_pts - row_sum[:, np.newaxis] * pts  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space at each point."""
    # tangent projection: g_t = g - (g . x) * x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_sphere(n, base_pts, sigma, seed):
    """Generate perturbed sphere points."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * sigma
    pts = base_pts + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_l_bfgs(start_pts):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        _, g = _energy_and_grad(pts)
        g_t = _tangent_project(pts, g)
        return g_t.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        best_pts = _project_to_sphere(result.x.reshape(N, 3))
        best_energy, _ = _energy_and_grad(best_pts)
        return best_pts, best_energy
    except Exception:
        return start_pts, float('inf')


def _gradient_descent_polish(pts0, max_time):
    """Polish with gradient descent using backtracking line search."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    lr = 0.01
    
    for _ in range(20000):
        if time.time() - start_time >= max_time:
            break
        
        g_tan = _tangent_project(pts, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-12:
            break
        
        # Backtracking line search
        step = lr
        new_pts = pts
        new_e = e
        for _ in range(30):
            candidate = pts - step * g_tan
            candidate = _project_to_sphere(candidate)
            cand_e, cand_g = _energy_and_grad(candidate)
            if cand_e < e:
                new_pts = candidate
                new_e = cand_e
                break
            step *= 0.5
        else:
            break
        
        pts = new_pts
        e = new_e
        g = _energy_and_grad(pts)[1]
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if e < best_e - 1e-10:
            lr = min(lr * 1.05, 0.1)
    
    return best_pts, best_e


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Best result so far
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Fibonacci sphere (baseline)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Random starts
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]:
        if time.time() - start_time > TIME_LIMIT - 40:
            break
        starts.append(_random_sphere(N, seed))
    
    # 3. Perturbed Fibonacci starts
    for sigma in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for seed in [1, 2, 3, 4]:
            if time.time() - start_time > TIME_LIMIT - 40:
                break
            starts.append(_perturbed_sphere(N, _fibonacci_sphere(N), sigma, seed))
    
    # 4. Scaled Fibonacci (stretch/compress)
    for scale in [0.8, 1.2]:
        fib = _fibonacci_sphere(N)
        # Apply a slight anisotropic scaling then re-project
        scaled = fib * np.array([scale, 1.0, 1.0])
        starts.append(_project_to_sphere(scaled))
    
    # Optimize from each start
    for i, start in enumerate(starts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        try:
            # First pass: L-BFGS
            pts, energy = _optimize_l_bfgs(start)
            
            # Second pass: gradient descent polish
            remaining = TIME_LIMIT - 10 - (time.time() - start_time)
            if remaining > 5:
                pts, energy = _gradient_descent_polish(pts, remaining)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
        except Exception:
            continue
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts


# EVOLVE-BLOCK-START
# (The entire logic is above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
