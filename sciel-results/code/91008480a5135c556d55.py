# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.transform import Rotation

N = 100

def _generate_bcc_base(n_points=100):
    """
    Generate a base configuration of n_points from a BCC lattice,
    selecting the points closest to the origin.
    """
    # Integer lattice points in [-2, 2]
    grid_int = np.array(np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3))).T.reshape(-1, 3)
    # Half-integer lattice points in [-1.5, 1.5]
    grid_half = np.array(np.meshgrid(np.arange(-1.5, 2, 1), np.arange(-1.5, 2, 1), np.arange(-1.5, 2, 1))).T.reshape(-1, 3)
    
    all_pts = np.vstack([grid_int, grid_half])
    
    # Center at origin (already centered)
    # Sort by distance to origin
    dists = np.linalg.norm(all_pts, axis=1)
    idx = np.argsort(dists)[:n_points]
    pts = all_pts[idx].copy()
    
    # Ensure exact centering (remove any drift from selection)
    pts -= np.mean(pts, axis=0)
    
    return pts

def _compute_radius(c):
    """
    Compute the largest common radius for given centers.
    c: (N, 3) array of centers.
    Returns radius r.
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair_dist, min_wall) / 2.0

def _force_directed_relax(centers, time_limit=60.0):
    """
    Relax the centers using force-directed dynamics.
    """
    start_time = time.time()
    c = centers.copy()
    n = len(c)
    
    # Learning rate schedule
    lr = 0.01
    decay = 0.999
    
    # Main loop
    while time.time() - start_time < time_limit:
        # Compute forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        # Force magnitude: 1/d^2
        # We want to maximize min distance, so repel strongly when close.
        # F_ij = (1/d^2) * u_ij
        inv_d2 = 1.0 / (dists_safe * dists_safe)
        np.fill_diagonal(inv_d2, 0.0)
        
        # Sum forces from all other particles
        # forces[i] += sum_j (1/d_ij^2) * (c_i - c_j) / d_ij
        # Note: diff[i,j] = c_i - c_j
        # Force on i from j is proportional to (c_i - c_j) / d_ij^3
        # Let's use standard Lennard-Jones-like or simple inverse square.
        # F_ij = k * (c_i - c_j) / d_ij^3
        # This is equivalent to gradient of 1/d_ij
        
        # Vectorized computation
        # diff is (N, N, 3)
        # dists is (N, N)
        # We need (N, N, 1) for dists to broadcast
        dists_3d = dists_safe[:, :, np.newaxis]
        
        # Force vector = diff / dist^3
        force_vec = diff / (dists_3d * dists_safe) # Wait, diff is c_i - c_j.
        # Gradient of 1/d is - (c_i - c_j) / d^3.
        # We want to move c_i away from c_j, so force should be in direction c_i - c_j.
        # So F_i = sum_j (c_i - c_j) / d_ij^3
        # This matches diff / d^3.
        
        # Sum over j (axis 1)
        forces += np.sum(force_vec, axis=1)
        
        # Wall repulsion
        # Distance to each wall
        # x_min = -0.5, x_max = 0.5
        # d_x_min = c[:, 0] - (-0.5) = c[:, 0] + 0.5
        # d_x_max = 0.5 - c[:, 0]
        # Force from x_min wall is in +x direction if close.
        # F_x_min = 1 / d_x_min^2 * (1, 0, 0)
        
        dx_min = c[:, 0] + 0.5
        dx_max = 0.5 - c[:, 0]
        dy_min = c[:, 1] + 0.5
        dy_max = 0.5 - c[:, 1]
        dz_min = c[:, 2] + 0.5
        dz_max = 0.5 - c[:, 2]
        
        # Safety clip for wall distances
        eps = 1e-6
        dx_min = np.clip(dx_min, eps, None)
        dx_max = np.clip(dx_max, eps, None)
        dy_min = np.clip(dy_min, eps, None)
        dy_max = np.clip(dy_max, eps, None)
        dz_min = np.clip(dz_min, eps, None)
        dz_max = np.clip(dz_max, eps, None)
        
        forces[:, 0] += (1.0 / dx_min**2) - (1.0 / dx_max**2)
        forces[:, 1] += (1.0 / dy_min**2) - (1.0 / dy_max**2)
        forces[:, 2] += (1.0 / dz_min**2) - (1.0 / dz_max**2)
        
        # Update positions
        c += lr * forces
        
        # Clip to box (hard constraint)
        c = np.clip(c, -0.5, 0.5)
        
        # Decay learning rate
        lr *= decay
        
        # Optional: Break if convergence is reached (force magnitude small)
        # if np.max(np.linalg.norm(forces, axis=1)) < 1e-6:
        #     break
        
    return c

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    Uses a rigid BCC lattice base, then relaxes using force-directed dynamics.
    """
    total_time_limit = 115.0 # Safety margin
    start_time = time.time()
    
    # 1. Generate base BCC lattice
    base_pts = _generate_bcc_base(N)
    
    # 2. Scale and center to fit in cube
    # The BCC points are in integer/half-integer coords.
    # Max extent is roughly 2.5.
    # We want them to fit in [-0.5, 0.5].
    # Let's scale by 0.15 initially?
    # Actually, let's find the max norm and scale to fit with some margin.
    max_norm = np.max(np.linalg.norm(base_pts, axis=1))
    # Target max radius ~ 0.4 (leaving 0.1 for walls)
    scale = 0.4 / max_norm
    centers = base_pts * scale
    
    # 3. Force-directed relaxation
    # Allocate most of the time to this
    remaining_time = total_time_limit - (time.time() - start_time)
    relaxation_time = max(5.0, remaining_time * 0.8)
    
    centers = _force_directed_relax(centers, time_limit=relaxation_time)
    
    # 4. Final check and return
    # Ensure centers are strictly inside (clip again just in case)
    centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return centers
# EVOLVE-BLOCK-END
