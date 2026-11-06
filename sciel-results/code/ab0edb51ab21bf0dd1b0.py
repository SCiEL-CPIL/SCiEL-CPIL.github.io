"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute log(1/d) energy and gradient for points on unit sphere."""
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid self-pairs and very small distances
    np.fill_diagonal(dist, 1.0)
    dist = np.maximum(dist, 1e-12)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = -np.sum(np.log(dist[mask]))
    
    # Gradient: d/dx_i [log(1/|x_i - x_j|)] = -(x_i - x_j)/|x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_dist2 = np.zeros_like(dist)
    inv_dist2[dist > 1e-12] = 1.0 / (dist[dist > 1e-12] ** 2)
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) / |x_i - x_j|^2
    # = sum_j inv_dist2[i,j] * (x_i - x_j)
    grad = np.zeros_like(pts)
    for i in range(n):
        # (x_i - x_j) for all j
        dx = pts[i] - pts  # shape (n, 3)
        # weight by inv_dist2[i, j]
        grad[i] = np.sum(inv_dist2[i, :][:, np.newaxis] * dx, axis=0)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _polish(pts, deadline):
    """Polish points using L-BFGS-B with time check."""
    n = len(pts)
    
    def obj_and_grad_flat(flat):
        pts = flat.reshape(n, 3)
        e, g = _energy_and_grad(pts)
        return e, g.ravel()
    
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Try L-BFGS-B with time limit
    try:
        result = minimize(
            obj_and_grad_flat,
            pts.ravel(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if result.success or result.fun < best_energy:
            new_pts = _project_to_sphere(result.x.reshape(n, 3))
            new_energy = _energy_and_grad(new_pts)[0]
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except Exception:
        pass
    
    return best_pts, best_energy

def _polish_refined(pts, deadline):
    """Refined polish with multiple L-BFGS-B passes."""
    n = len(pts)
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    for _ in range(3):
        if time.time() > deadline - 5:
            break
        
        def obj_and_grad_flat(flat):
            pts_local = flat.reshape(n, 3)
            e, g = _energy_and_grad(pts_local)
            return e, g.ravel()
        
        try:
            result = minimize(
                obj_and_grad_flat,
                best_pts.ravel(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if result.fun < best_energy:
                new_pts = _project_to_sphere(result.x.reshape(n, 3))
                new_energy = _energy_and_grad(new_pts)[0]
                if new_energy < best_energy:
                    best_pts = new_pts
                    best_energy = new_energy
        except Exception:
            break
    
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(fib_pts)
    
    # 2. Random points (seeded for reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(5):
        rand_pts = rng.randn(N, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        initializations.append(rand_pts)
    
    # 3. Spherical coordinates with jitter
    for seed in range(3):
        rng2 = np.random.RandomState(100 + seed)
        theta = np.sort(rng2.uniform(0, np.pi, N))
        phi = rng2.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        initializations.append(pts)
    
    best_pts = None
    best_energy = float('inf')
    
    for init_pts in initializations:
        if time.time() > deadline - 10:
            break
        
        # Quick polish
        polished_pts, energy = _polish(init_pts, deadline)
        if energy < best_energy:
            best_energy = energy
            best_pts = polished_pts
        
        # Refined polish if time permits
        if time.time() < deadline - 15:
            refined_pts, refined_energy = _polish_refined(polished_pts, deadline)
            if refined_energy < best_energy:
                best_energy = refined_energy
                best_pts = refined_pts
    
    # Final check
    if best_pts is None:
        best_pts = fib_pts
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points too close
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
