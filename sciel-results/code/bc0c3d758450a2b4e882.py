"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(centers):
    """Compute the minimum radius for non-overlapping spheres inside [-0.5, 0.5]^3."""
    # Pairwise distances
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf so it doesn't affect min
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r


def _initial_configuration():
    """Generate initial configuration using a grid with some perturbation."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.01, size=centers.shape)
    # Clamp to cube
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _local_refine(centers, max_iters=5000, step_size=0.01):
    """Local optimization: push overlapping spheres apart and keep in bounds."""
    centers = centers.copy()
    rng = np.random.default_rng(123)
    
    for it in range(max_iters):
        # Compute forces
        n = len(centers)
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)  # avoid division by zero
        
        # Only consider pairs that are too close
        target_dist = 0.27  # slightly more than 2*0.135
        mask = dists < target_dist
        mask[np.diag_indices_from(mask)] = False
        
        for i in range(n):
            for j in range(i+1, n):
                if mask[i, j]:
                    d = diff[i, j]
                    dist = dists[i, j]
                    if dist < 1e-10:
                        d = rng.normal(0, 0.001, 3)
                        dist = np.sqrt(np.sum(d**2))
                    # Push apart
                    force = d / dist * (target_dist - dist)
                    forces[i] += force
                    forces[j] -= force
        
        # Wall forces: push inward if too close to wall
        for i in range(n):
            for k in range(3):
                margin = 0.135
                if centers[i, k] > 0.5 - margin:
                    forces[i, k] -= (centers[i, k] - (0.5 - margin))
                elif centers[i, k] < -0.5 + margin:
                    forces[i, k] += ((-0.5 + margin) - centers[i, k])
        
        # Update positions with annealing step
        centers += step_size * forces
        centers = np.clip(centers, -0.499, 0.499)
        
        # Decay step size
        step_size *= 0.999
    
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_centers = _initial_configuration()
    best_radius = _compute_radius(best_centers)
    
    # Multiple restarts with different initial conditions
    rng = np.random.default_rng()
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Create varied initial configurations
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        
        # Different perturbation strategies
        if restart % 3 == 0:
            centers += rng.normal(0, 0.02, size=centers.shape)
        elif restart % 3 == 1:
            centers += rng.normal(0, 0.005, size=centers.shape)
        else:
            # Swap some points
            idx = rng.choice(N, size=5, replace=False)
            centers[idx] = rng.uniform(-0.4, 0.4, size=(len(idx), 3))
        
        centers = np.clip(centers, -0.49, 0.49)
        
        # Local refinement
        centers = _local_refine(centers, max_iters=3000, step_size=0.008)
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius < 0.1:  # Sanity check
        # Fall back to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
