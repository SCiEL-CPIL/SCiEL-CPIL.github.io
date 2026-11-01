"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
import random

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    # Wall constraint: distance from each center to the nearest wall
    wall_dists = np.min(np.abs(centers - 0.5) - np.min(np.abs(centers - 0.5), axis=1)[:, None], axis=1)
    # Actually, for cube [-0.5, 0.5]^3, the distance to the boundary for point c is:
    # min(0.5 - c[0], 0.5 - c[1], 0.5 - c[2], c[0] + 0.5, c[1] + 0.5, c[2] + 0.5)
    # which is min(0.5 - |c_k|) over k for each coordinate, then min over coordinates
    dists_to_wall = np.min(0.5 - np.abs(centers), axis=1)
    wall_radius = np.min(dists_to_wall)
    
    # Pairwise distances
    if len(centers) < 2:
        pair_radius = np.inf
    else:
        d = cdist(centers, centers)
        np.fill_diagonal(d, np.inf)
        pair_radius = np.min(d) / 2.0
    
    return min(wall_radius, pair_radius)

def _soft_min(x, t=0.01):
    """Soft-min approximation using log-sum-exp trick."""
    x = np.asarray(x)
    return -np.log(np.sum(np.exp(-x / t))) * t

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Multiple random restarts with local optimization
    np.random.seed(42)
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        # Initialize with random points in the cube
        centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Local optimization using gradient-free method (Nelder-Mead style)
        # We'll use a simple iterative repulsion-based approach
        step_size = 0.05
        for it in range(500):
            if time.time() > deadline - 5.0:
                break
            
            # Compute all pairwise distances and wall distances
            d = cdist(centers, centers)
            np.fill_diagonal(d, np.inf)
            min_pair_dist = np.min(d)
            min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
            
            dists_to_wall = np.min(0.5 - np.abs(centers), axis=1)
            min_wall_idx = np.argmin(dists_to_wall)
            min_wall_dist = dists_to_wall[min_wall_idx]
            
            # Determine the bottleneck
            if min_pair_dist / 2.0 < min_wall_dist:
                # Bottleneck is a pair of spheres
                i, j = min_pair_idx
                # Push i and j apart
                diff = centers[i] - centers[j]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    diff = np.random.randn(3)
                    norm = np.linalg.norm(diff)
                diff /= norm
                centers[i] += step_size * diff * 0.5
                centers[j] -= step_size * diff * 0.5
            else:
                # Bottleneck is a wall
                i = min_wall_idx
                # Push sphere i toward the center
                center_vec = -centers[i]
                norm = np.linalg.norm(center_vec)
                if norm < 1e-10:
                    center_vec = np.random.randn(3)
                    norm = np.linalg.norm(center_vec)
                center_vec /= norm
                centers[i] += step_size * center_vec
            
            # Clamp to cube
            centers = np.clip(centers, -0.499, 0.499)
            
            # Decrease step size gradually
            step_size *= 0.995
        
        # Evaluate this configuration
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final polish: high-precision local optimization on the best configuration
    if best_centers is not None and time.time() < deadline - 2.0:
        centers = best_centers.copy()
        step_size = 0.001
        for it in range(2000):
            if time.time() > deadline - 1.0:
                break
            
            d = cdist(centers, centers)
            np.fill_diagonal(d, np.inf)
            min_pair_dist = np.min(d)
            min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
            
            dists_to_wall = np.min(0.5 - np.abs(centers), axis=1)
            min_wall_idx = np.argmin(dists_to_wall)
            min_wall_dist = dists_to_wall[min_wall_idx]
            
            if min_pair_dist / 2.0 < min_wall_dist:
                i, j = min_pair_idx
                diff = centers[i] - centers[j]
                norm = np.linalg.norm(diff)
                if norm < 1e-12:
                    diff = np.random.randn(3)
                    norm = np.linalg.norm(diff)
                diff /= norm
                centers[i] += step_size * diff * 0.5
                centers[j] -= step_size * diff * 0.5
            else:
                i = min_wall_idx
                center_vec = -centers[i]
                norm = np.linalg.norm(center_vec)
                if norm < 1e-12:
                    center_vec = np.random.randn(3)
                    norm = np.linalg.norm(center_vec)
                center_vec /= norm
                centers[i] += step_size * center_vec
            
            centers = np.clip(centers, -0.499999, 0.499999)
            step_size *= 0.998
        
        radius = _compute_radius(centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
