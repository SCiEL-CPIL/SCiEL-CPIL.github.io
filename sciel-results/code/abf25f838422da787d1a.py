"""Riemannian mirror-descent for logarithmic energy minimization on the sphere."""
import time
import math
import numpy as np

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
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sum = inv_dist_sq.sum(axis=1)
    W_pts = inv_dist_sq @ pts
    grad = W_pts - row_sum[:, np.newaxis] * pts
    
    return energy, grad


def _tangent_project(pts, grad):
    """Project gradient onto tangent space at each point."""
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts


def _exponential_map(pts, tangent_vecs):
    """
    Apply the exponential map on the unit sphere.
    For a point x on S^2 and tangent vector v at x:
    exp_x(v) = x*cos(|v|) + (v/|v|)*sin(|v|)
    When |v| is small, use Taylor expansion.
    """
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.maximum(norms, 1e-15)
    
    # For small angles, use Taylor: cos(t) ≈ 1 - t^2/2, sin(t)/t ≈ 1 - t^2/6
    small_mask = norms < 1e-8
    cos_t = np.cos(norms_safe)
    sin_t_over_t = np.sin(norms_safe) / norms_safe
    
    # Handle small norms with Taylor
    if np.any(small_mask):
        t2 = norms_safe[small_mask, 0] ** 2
        cos_t[small_mask, 0] = 1.0 - t2 / 2.0
        sin_t_over_t[small_mask, 0] = 1.0 - t2 / 6.0
    
    # exp_x(v) = x * cos(|v|) + v * (sin(|v|)/|v|)
    result = pts * cos_t + tangent_vecs * sin_t_over_t
    
    # Re-normalize to ensure exact unit length
    result_norms = np.linalg.norm(result, axis=1, keepdims=True)
    result_norms = np.maximum(result_norms, 1e-15)
    return result / result_norms


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


def _riemannian_mirror_descent(pts0, max_time, initial_lr=0.01):
    """
    Riemannian mirror-descent using exponential map.
    Operates directly on the sphere manifold:
    1. Compute Riemannian gradient (tangent projection of Euclidean gradient)
    2. Take step in tangent space
    3. Map back to manifold via exponential map
    4. Adaptive step size with backtracking
    """
    start_time = time.time()
    pts = pts0.copy()
    e, g = _energy_and_grad(pts)
    best_pts = pts.copy()
    best_e = e
    
    lr = initial_lr
    min_lr = 1e-10
    max_lr = 0.1
    
    for it in range(50000):
        if time.time() - start_time >= max_time:
            break
        
        # Riemannian gradient (tangent projection)
        g_tan = _tangent_project(pts, g)
        g_norm = np.linalg.norm(g_tan)
        
        if g_norm < 1e-12:
            break
        
        # Backtracking line search with exponential map
        step = lr
        new_pts = pts
        new_e = e
        new_g = g
        
        for _ in range(40):
            # Step in tangent space
            tangent_step = -step * g_tan
            # Map back to manifold
            candidate = _exponential_map(pts, tangent_step)
            cand_e, cand_g = _energy_and_grad(candidate)
            
            if cand_e < e - 1e-14:
                new_pts = candidate
                new_e = cand_e
                new_g = cand_g
                break
            step *= 0.5
        
        if new_pts is pts:
            # No improvement even with tiny step; try smaller lr
            lr = max(lr * 0.5, min_lr)
            if lr <= min_lr:
                break
            continue
        
        pts = new_pts
        e = new_e
        g = new_g
        
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if e < best_e - 1e-10:
            lr = min(lr * 1.1, max_lr)
    
    return best_pts, best_e


def _lbfgs_polish(pts0, max_time):
    """
    Quick L-BFGS polish using scipy, operating in ambient space
    with sphere projection.
    """
    from scipy.optimize import minimize
    
    x0 = pts0.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        e, _ = _energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts = pts / norms
        _, g = _energy_and_grad(pts)
        g_t = _tangent_project(pts, g)
        return g_t.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=jac,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 30}
        )
        best_pts = result.x.reshape(N, 3)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        best_pts = best_pts / norms
        best_energy, _ = _energy_and_grad(best_pts)
        return best_pts, best_energy
    except Exception:
        return pts0, float('inf')


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
    
    # 2. Random starts with varied seeds
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066, 7077, 8088]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        starts.append(_random_sphere(N, seed))
    
    # 3. Perturbed Fibonacci starts with varied noise levels
    for sigma in [0.03, 0.07, 0.15, 0.3, 0.5]:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() - start_time > TIME_LIMIT - 60:
                break
            starts.append(_perturbed_sphere(N, _fibonacci_sphere(N), sigma, seed))
    
    # 4. Anisotropically scaled Fibonacci
    for sx, sy, sz in [(0.8, 1.0, 1.0), (1.2, 1.0, 1.0), (1.0, 0.8, 1.0), (1.0, 1.2, 1.0)]:
        fib = _fibonacci_sphere(N)
        scaled = fib * np.array([sx, sy, sz])
        norms = np.linalg.norm(scaled, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        starts.append(scaled / norms)
    
    # Optimize from each start
    for i, start in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        try:
            # Phase 1: Riemannian mirror-descent (main optimization)
            remaining_md = min(30.0, TIME_LIMIT - 25 - elapsed)
            if remaining_md > 2:
                pts, energy = _riemannian_mirror_descent(start, remaining_md, initial_lr=0.005)
            else:
                pts, energy = start, _energy_and_grad(start)[0]
            
            # Phase 2: L-BFGS polish
            elapsed2 = time.time() - start_time
            remaining_lbfgs = min(15.0, TIME_LIMIT - 10 - elapsed2)
            if remaining_lbfgs > 3:
                pts2, energy2 = _lbfgs_polish(pts, remaining_lbfgs)
                if energy2 < energy:
                    pts, energy = pts2, energy2
            
            # Phase 3: Final Riemannian polish with small lr
            elapsed3 = time.time() - start_time
            remaining_final = TIME_LIMIT - 5 - elapsed3
            if remaining_final > 2:
                pts3, energy3 = _riemannian_mirror_descent(pts, remaining_final, initial_lr=0.001)
                if energy3 < energy:
                    pts, energy = pts3, energy3
            
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
                
        except Exception:
            continue
    
    # Final validation
    best_pts = best_pts.copy()
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
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
