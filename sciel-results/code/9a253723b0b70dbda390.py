"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dist[mask]
    d = np.maximum(d, 1e-15)
    terms = np.log(1.0 / d)
    return math.fsum(terms)

def _repulsive_dynamics(points, steps=5000, alpha_start=0.01, alpha_end=0.00001):
    """
    Minimize log energy using repulsive force dynamics with cooling.
    Force is repulsive: F_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    pts = points.copy()
    
    for step in range(steps):
        # Exponential cooling
        alpha = alpha_start * (alpha_end / alpha_start) ** (step / max(1, steps - 1))
        
        # Calculate forces
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Force on i is sum_j (pts[i] - pts[j]) / dist_sq[i, j]
        # diff[i, j] = pts[i] - pts[j]
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project forces to tangent plane
        dot_products = np.sum(forces * pts, axis=1, keepdims=True)
        tangent_forces = forces - dot_products * pts
        
        # Update points
        pts = pts + alpha * tangent_forces
        
        # Normalize to sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
    return pts

def construct():
    best_points = None
    best_energy = float('inf')
    
    # Run multiple restarts
    # Use a fixed base seed for reproducibility, but vary the offset
    base_seed = 42
    
    for restart in range(8):
        np.random.seed(base_seed + restart)
        
        # Random initialization
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Run dynamics
        t_start = time.time()
        candidate = _repulsive_dynamics(pts, steps=4000, alpha_start=0.02, alpha_end=0.00005)
        t_end = time.time()
        
        # Check validity
        if not np.all(np.isfinite(candidate)):
            continue
            
        norms = np.linalg.norm(candidate, axis=1)
        if np.any(norms < 1e-15):
            continue
            
        diff = candidate[:, np.newaxis, :] - candidate[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        if np.min(dist) < 1e-9:
            continue
            
        energy = _log_energy(candidate)
        
        if energy < best_energy:
            best_energy = energy
            best_points = candidate
            
        # Time check
        if time.time() > 100:
            break
            
    if best_points is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        best_points = best_points / norms
        
    return best_points
# EVOLVE-BLOCK-END
