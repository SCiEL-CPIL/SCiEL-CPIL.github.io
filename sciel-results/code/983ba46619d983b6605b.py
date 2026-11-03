"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = pts.shape[0]
    # Compute all pairwise distances
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal and lower triangle to get i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    
    if np.any(d < 1e-9):
        return float('inf')
        
    # log(1/d) = -log(d)
    terms = -np.log(d)
    return math.fsum(terms.tolist())

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin before 120s kill
    
    # Start with golden-angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _log_energy(best_pts)
    
    # Vectorized Gradient Descent
    # Gradient of E = sum log(1/d_ij) wrt p_i is sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    # We minimize E, so we move in direction -grad(E).
    # -grad(E)_i = sum_{j != i} (p_j - p_i) / ||p_i - p_j||^2
    
    step_size = 0.1
    
    # Optimization loop
    while time.time() < deadline:
        # Compute pairwise differences and distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
        dists = np.linalg.norm(diff, axis=2)  # (N, N)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists, 1.0)
        
        # Weight matrix: 1 / dist^2
        # We want to compute sum_j (p_j - p_i) / d_ij^2
        # Note: diff[i,j] = p_i - p_j. So p_j - p_i = -diff[i,j].
        # grad_i = sum_j (p_i - p_j) / d_ij^2 = sum_j diff[i,j] / d_ij^2
        # We want to move in -grad direction.
        
        inv_d2 = 1.0 / (dists * dists)
        inv_d2[np.diag_indices(N)] = 0.0
        
        # Gradient at i is sum_j diff[i,j] * inv_d2[i,j]
        # diff is (N, N, 3), inv_d2 is (N, N)
        # grad = np.einsum('ijk,ij->ik', diff, inv_d2)
        grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Project gradient to tangent space
        # proj_grad_i = grad_i - (grad_i . p_i) p_i
        dots = np.sum(grad * pts, axis=1)
        proj_grad = grad - dots[:, np.newaxis] * pts
        
        # Update
        pts = pts - step_size * proj_grad
        pts = _normalize(pts)
        
        # Compute energy
        energy = _log_energy(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Decay step size
        step_size *= 0.995
        
        # If step size becomes too small, break or restart? 
        # For now, just let it run. The vectorized ops are fast.
        
    return best_pts
# EVOLVE-BLOCK-END
