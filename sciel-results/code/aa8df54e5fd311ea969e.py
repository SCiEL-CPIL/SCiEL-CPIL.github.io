"""Optimized logarithmic energy minimization of 150 points on the sphere."""
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
    # Compute pairwise squared distances efficiently
    # |x_i - x_j|^2 = |x_i|^2 + |x_j|^2 - 2*x_i.x_j = 2 - 2*x_i.x_j for unit vectors
    dot = pts @ pts.T  # (N, N)
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero / log(0)
    
    # Energy: sum over i<j of -log(dist_ij) = -0.5 * sum_{i!=j} log(dist_ij)
    # = -0.5 * sum_{i!=j} 0.5*log(dist_sq_ij) = -0.25 * sum_{i!=j} log(dist_sq_ij)
    # But we need sum_{i<j}, so: energy = -sum_{i<j} log(sqrt(dist_sq)) = -0.5*sum_{i<j} log(dist_sq)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    # Gradient: dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    # Using dist_sq directly (no sqrt needed)
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


def _optimize_l_bfgs(start_pts, max_time=30.0):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = start_pts.flatten()
    t0 = time.time()
    
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
            options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 40}
        )
        best_pts = _project_to_sphere(result.x.reshape(N, 3))
        best_energy, _ = _energy_and_grad(best_pts)
        return best_pts, best_energy
    except Exception:
        return start_pts, float('inf')


def _gradient_descent_polish(pts0, max_time):
    """Polish with gradient descent using backtracking line search and momentum."""
    start_time = time.time()
    pts = _project_to_sphere(pts0.copy())
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    lr = 0.01
    momentum = np.zeros_like(pts)
    beta = 0.9
    
    for _ in range(50000):
        if time.time() - start_time >= max_time:
            break
        
        g_tan = _tangent_project(pts, g)
        g_norm = np.linalg.norm(g_tan)
        if g_norm < 1e-13:
            break
        
        # Momentum update
        momentum = beta * momentum - lr * g_tan
        
        # Backtracking line search
        step = 1.0
        new_pts = pts
        new_e = e
        found = False
        for _ in range(25):
            candidate = pts + step * momentum
            candidate = _project_to_sphere(candidate)
            cand_e, _ = _energy_and_grad(candidate)
            if cand_e < e:
                new_pts = candidate
                new_e = cand_e
                found = True
                break
            step *= 0.5
        if not found:
            # Try plain gradient step
            step = lr * 0.1
            candidate = pts - step * g_tan
            candidate = _project_to_sphere(candidate)
            cand_e, _ = _energy_and_grad(candidate)
            if cand_e < e:
                new_pts = candidate
                new_e = cand_e
                found = True
                momentum = -step * g_tan
            else:
                break
        
        if not found:
            break
        
        pts = new_pts
        e = new_e
        g = _energy_and_grad(pts)[1]
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            # Slightly increase lr if making good progress
            if best_e < e + 1e-8:
                lr = min(lr * 1.02, 0.05)
    
    return best_pts, best_e


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Best result so far
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Generate diverse starting configurations - prioritize quality
    starts = []
    
    # 1. Fibonacci sphere (baseline)
    starts.append(_fibonacci_sphere(N))
    
    # 2. A few good random starts
    for seed in [42, 123, 789, 2022, 5055]:
        starts.append(_random_sphere(N, seed))
    
    # 3. Perturbed Fibonacci starts (moderate perturbations)
    for sigma in [0.1, 0.2, 0.3]:
        for seed in [1, 2]:
            starts.append(_perturbed_sphere(N, _fibonacci_sphere(N), sigma, seed))
    
    # 4. Slightly anisotropic scaling
    fib = _fibonacci_sphere(N)
    for scale in [0.85, 1.15]:
        scaled = fib * np.array([scale, 1.0, 1.0])
        starts.append(_project_to_sphere(scaled))
    
    # Optimize from each start
    for i, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 25:
            break
        
        try:
            # First pass: L-BFGS with time limit
            lbfgs_time = min(20.0, TIME_LIMIT - 20 - elapsed)
            pts, energy = _optimize_l_bfgs(start, max_time=lbfgs_time)
            
            # Second pass: gradient descent polish with remaining time
            remaining = TIME_LIMIT - 15 - (time.time() - start_time)
            if remaining > 3:
                pts, energy = _gradient_descent_polish(pts, remaining)
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Final high-precision polish on the best result
    remaining = TIME_LIMIT - 8 - (time.time() - start_time)
    if remaining > 2:
        try:
            # Tight L-BFGS polish
            x0 = best_pts.flatten()
            
            def obj_final(x):
                pts = _project_to_sphere(x.reshape(N, 3))
                e, _ = _energy_and_grad(pts)
                return e
            
            def jac_final(x):
                pts = _project_to_sphere(x.reshape(N, 3))
                _, g = _energy_and_grad(pts)
                g_t = _tangent_project(pts, g)
                return g_t.flatten()
            
            result = minimize(
                obj_final, x0, jac=jac_final,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 30}
            )
            polished = _project_to_sphere(result.x.reshape(N, 3))
            pol_e, _ = _energy_and_grad(polished)
            if pol_e < best_energy:
                best_energy = pol_e
                best_pts = polished
        except Exception:
            pass
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    dot = best_pts @ best_pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts


# EVOLVE-BLOCK-START
# (The entire logic is above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
