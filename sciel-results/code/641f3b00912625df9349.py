"""Optimized logarithmic energy minimization for 150 points on the unit sphere."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE_S = 115.0  # safety margin before 120s limit


def _energy_and_grad(pts):
    """Compute log energy sum_{i<j} log(1/|x_i-x_j|) and gradient.
    
    Energy = -sum_{i<j} log(|x_i - x_j|)
    Gradient wrt x_k: sum_{j!=k} (x_k - x_j) / |x_k - x_j|^2
    """
    n = pts.shape[0]
    # Pairwise distances
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist[i,j])
    # Upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point k, grad_k = sum_{j!=k} (x_k - x_j) / |x_k - x_j|^2
    # = sum_{j!=k} diff[k,j] / dist[k,j]^2
    # diff[k,j] / dist[k,j]^2 = diff[k,j] * (1/dist[k,j]^2)
    inv_dist2 = np.zeros_like(dist)
    np.fill_diagonal(inv_dist2, 0.0)
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[k] = sum_j diff[k,j] * inv_dist2[k,j]
    grad = np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project(pts):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _halton_sphere():
    """Generate Halton sequence points on the unit sphere."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r
    
    pts = np.zeros((N, 3))
    for i in range(N):
        u = halton(i + 1, 2)
        v = halton(i + 1, 3)
        w = halton(i + 1, 5)
        # Use inverse CDF for uniform on sphere
        theta = 2.0 * np.pi * u
        cos_phi = 2.0 * v - 1.0
        sin_phi = np.sqrt(1.0 - cos_phi * cos_phi)
        pts[i] = [sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi]
    return pts


def _random_sphere(seed):
    """Generate random points on the unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _optimize_from_start(pts0, time_budget, tol=1e-12, maxiter=2000):
    """Run constrained optimization from a starting point.
    
    Uses gradient descent with projection, with adaptive step size.
    """
    pts = pts0.copy().astype(np.float64)
    pts = _project(pts)
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Initial step size
    step = 0.01
    min_step = 1e-10
    max_step = 0.5
    
    n_iter = 0
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        energy, grad = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step size with backtracking
        accepted = False
        for _ in range(30):
            new_pts = pts - step * grad
            new_pts = _project(new_pts)
            new_energy, _ = _energy_and_grad(new_pts)
            
            if new_energy < energy - 1e-15:
                pts = new_pts
                accepted = True
                break
            step *= 0.5
        
        if not accepted:
            # Try smaller step
            step = max(step * 0.1, min_step)
            if step < min_step:
                break
        
        # Increase step if we made good progress
        if accepted and step < max_step:
            step = min(step * 1.1, max_step)
        
        n_iter += 1
        
        # Periodic check
        if n_iter % 50 == 0:
            pass  # Could add more sophisticated convergence checks
    
    return best_pts, best_energy


def _polish(pts, time_budget):
    """High-precision polish using scipy minimize with tighter tolerance."""
    # Flatten for scipy
    x0 = pts.flatten()
    
    def fun_and_grad(x):
        p = x.reshape(N, 3)
        p = _project(p)
        e, g = _energy_and_grad(p)
        return e, g.flatten()
    
    try:
        res = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.success or res.fun < fun_and_grad(x0)[0]:
            pts_res = _project(res.x.reshape(N, 3))
            e_res, _ = _energy_and_grad(pts_res)
            return pts_res, e_res
    except Exception:
        pass
    
    return pts, _energy_and_grad(pts)[0]


def construct():
    t0 = time.time()
    
    # Generate multiple diverse starting points
    starts = []
    starts.append(_fibonacci_sphere())
    starts.append(_halton_sphere())
    
    # Add several random starts
    for seed in [42, 123, 456, 789, 1011, 2023]:
        starts.append(_random_sphere(seed))
    
    # Optimize each start, allocating time budget
    best_pts = None
    best_energy = np.inf
    
    n_starts = len(starts)
    time_per_start = (DEADLINE_S - (time.time() - t0)) / n_starts
    # Reserve 20% for polishing
    time_per_opt = time_per_start * 0.8
    polish_time = time_per_start * 0.2
    
    for i, start in enumerate(starts):
        elapsed = time.time() - t0
        remaining = DEADLINE_S - elapsed
        
        if remaining < 5.0:
            break
        
        # Allocate remaining time fairly
        time_left_for_opt = min(time_per_opt, remaining * 0.7 / (n_starts - i))
        time_left_for_opt = max(time_left_for_opt, 1.0)
        
        pts, energy = _optimize_from_start(start, time_left_for_opt)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
    
    # Final polish on the best result
    elapsed = time.time() - t0
    remaining = DEADLINE_S - elapsed
    
    if best_pts is not None and remaining > 3.0:
        polish_budget = min(remaining * 0.8, 30.0)
        best_pts, best_energy = _polish(best_pts, polish_budget)
    
    # Ensure validity
    if best_pts is None:
        best_pts = _fibonacci_sphere()
    
    # Final projection and validation
    best_pts = _project(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Nudge point j slightly
                noise = np.random.default_rng(j).normal(size=3) * 1e-6
                best_pts[j] = best_pts[j] + noise
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
