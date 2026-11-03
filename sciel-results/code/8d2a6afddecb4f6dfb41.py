"""Optimized logarithmic energy minimization for 100 points on the sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # Safety margin before 150s kill


def _energy_and_grad(pts):
    """Compute log(1/d) energy and its gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|xi - xj|) = -sum_{i<j} log(|xi - xj|)
    Gradient w.r.t. xi: sum_{j!=i} (xi - xj) / |xi - xj|^2
    """
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # Avoid zero on diagonal
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each point i, grad_i = sum_{j!=i} (xi - xj) / |xi - xj|^2
    # diff[i,j] = xi - xj, so (xi-xj)/|xi-xj|^2 = diff[i,j] / dist_sq[i,j]
    grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i != j:
                d2 = dist_sq[i, j]
                if d2 > 1e-30:
                    grad[i] += diff[i, j] / d2
    return energy, grad


def _energy_fast(pts):
    """Faster energy computation using vectorized operations."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu]))


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _gradient_flow_anneal(pts, n_steps=200, lr_start=0.01, lr_end=0.0001, deadline=None):
    """Gradient flow with annealing learning rate."""
    current = pts.copy()
    best_energy = _energy_fast(current)
    best_pts = current.copy()
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Anneal learning rate
        t = step / max(n_steps - 1, 1)
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute gradient
        n = current.shape[0]
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Gradient: sum_j (xi - xj) / |xi - xj|^2
        grad = np.zeros_like(current)
        for i in range(n):
            for j in range(n):
                if i != j:
                    d2 = dist_sq[i, j]
                    if d2 > 1e-30:
                        grad[i] += diff[i, j] / d2
        
        # Update
        current = current - lr * grad
        current = _project_to_sphere(current)
        
        energy = _energy_fast(current)
        if energy < best_energy:
            best_energy = energy
            best_pts = current.copy()
    
    return best_pts, best_energy


def _local_optimize(pts, deadline=None):
    """L-BFGS local optimization from a starting point."""
    if deadline is not None and time.time() > deadline:
        return pts, _energy_fast(pts)
    
    def objective(x):
        pts_local = _project_to_sphere(x.reshape(N, 3))
        return _energy_fast(pts_local)
    
    def gradient(x):
        pts_local = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(pts_local)
        return g.flatten()
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
        best_energy = _energy_fast(best_pts)
        return best_pts, best_energy
    except Exception:
        return pts, _energy_fast(pts)


def _polish(pts, deadline=None):
    """High-precision polish with tiny perturbations and tighter tolerances."""
    if deadline is not None and time.time() > deadline:
        return pts, _energy_fast(pts)
    
    best_pts = pts.copy()
    best_energy = _energy_fast(best_pts)
    
    # Try multiple tiny perturbations
    rng = np.random.default_rng(42)
    for trial in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        # Tiny perturbation
        noise = rng.standard_normal((N, 3)) * 1e-4
        perturbed = _project_to_sphere(best_pts + noise)
        
        def objective(x):
            pts_local = _project_to_sphere(x.reshape(N, 3))
            return _energy_fast(pts_local)
        
        def gradient(x):
            pts_local = _project_to_sphere(x.reshape(N, 3))
            e, g = _energy_and_grad(pts_local)
            return g.flatten()
        
        try:
            res = minimize(
                objective,
                perturbed.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 100}
            )
            polished = _project_to_sphere(res.x.reshape(N, 3))
            energy = _energy_fast(polished)
            if energy < best_energy:
                best_energy = energy
                best_pts = polished.copy()
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_pts = None
    best_energy = float('inf')
    
    # Strategy 1: Start from Fibonacci spiral
    fib_pts = _fibonacci_spiral(N)
    
    # Gradient flow annealing from Fibonacci start
    annealed_pts, anneal_energy = _gradient_flow_anneal(
        fib_pts, n_steps=100, lr_start=0.005, lr_end=0.00005, deadline=deadline
    )
    if anneal_energy < best_energy:
        best_energy = anneal_energy
        best_pts = annealed_pts.copy()
    
    # Local optimization from annealed point
    if time.time() < deadline:
        opt_pts, opt_energy = _local_optimize(annealed_pts, deadline=deadline)
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    # Strategy 2: Try a few random restarts with annealing
    n_restarts = 3
    for r in range(n_restarts):
        if time.time() > deadline - 30:  # Leave time for polish
            break
        
        seed = 1000 + r * 7
        rand_pts = _random_points(N, seed=seed)
        
        # Shorter annealing for random starts
        annealed_pts_r, anneal_energy_r = _gradient_flow_anneal(
            rand_pts, n_steps=80, lr_start=0.003, lr_end=0.00003, deadline=deadline
        )
        if anneal_energy_r < best_energy:
            best_energy = anneal_energy_r
            best_pts = annealed_pts_r.copy()
        
        # Local optimization
        if time.time() < deadline - 20:
            opt_pts_r, opt_energy_r = _local_optimize(annealed_pts_r, deadline=deadline)
            if opt_energy_r < best_energy:
                best_energy = opt_energy_r
                best_pts = opt_pts_r.copy()
    
    # Final polish
    if best_pts is not None and time.time() < deadline - 10:
        polished_pts, polished_energy = _polish(best_pts, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts.copy()
    
    # Ensure validity
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Final check: no points closer than 1e-9
    dists = np.sqrt(np.sum((best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Ensure all points are on unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    
    return best_pts
