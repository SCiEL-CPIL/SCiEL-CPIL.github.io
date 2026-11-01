"""Optimized logarithmic energy minimizer using repulsive force + annealing."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math


def _log_energy(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|xi-xj|)."""
    # Vectorized pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Upper triangle
    iu = np.triu_indices(N, k=1)
    d = np.sqrt(d2[iu])
    if np.any(d < 1e-12):
        return float('inf')
    return float(np.sum(np.log(1.0 / d)))


def _compute_forces(pts):
    """Compute repulsive forces: force on i = sum_{j!=i} (xi-xj)/|xi-xj|^2"""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Set diagonal to 1 to avoid division by zero
    np.fill_diagonal(d2, 1.0)
    # Force on i from j: diff[i,j] / d2[i,j]
    inv_d2 = 1.0 / d2  # (N, N)
    # Zero out diagonal
    inv_d2[np.diag_indices(N)] = 0.0
    forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (N, 3)
    return forces


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def construct():
    rng = np.random.RandomState(42)
    best_pts = None
    best_energy = float('inf')
    
    n_restarts = 4
    n_iters = 4000
    
    for restart in range(n_restarts):
        # Initialize with random points
        pts = rng.randn(N, 3)
        pts = _project_to_sphere(pts)
        
        # Annealing schedule
        step_size = 0.08
        cool_rate = 0.999
        min_step = 1e-7
        
        for it in range(n_iters):
            forces = _compute_forces(pts)
            
            # Update positions
            new_pts = pts + step_size * forces
            new_pts = _project_to_sphere(new_pts)
            
            # Check energy periodically
            if it % 100 == 0:
                e = _log_energy(new_pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = new_pts.copy()
            
            pts = new_pts
            step_size = max(step_size * cool_rate, min_step)
    
    # Final polish
    if best_pts is not None:
        pts = best_pts.copy()
        for it in range(2000):
            forces = _compute_forces(pts)
            step = max(0.002 * (1 - it / 2000), 1e-8)
            new_pts = pts + step * forces
            new_pts = _project_to_sphere(new_pts)
            pts = new_pts
        e = _log_energy(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    return best_pts if best_pts is not None else _project_to_sphere(rng.randn(N, 3))
# EVOLVE-BLOCK-END
