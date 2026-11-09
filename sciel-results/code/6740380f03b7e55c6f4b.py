"""Morse cluster global minima search with basin hopping."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# We aim to finish by 290s to be safe, leaving 10s buffer for overhead/evaluation
_DEADLINE = _START_TIME + 290.0


def _time_remaining():
    return _DEADLINE - time.time()


def _check_validity(positions, min_dist=0.5):
    """Check if positions are valid (no overlaps)."""
    if len(positions) < 2:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_dist


def _fibonacci_sphere(n, radius, rng):
    """
    Generate n points on a sphere of given radius using Fibonacci lattice.
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    i = i + 0.5
    cos_theta = 1.0 - (i / n)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    
    # Random rotation to avoid alignment with axes
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    theta = i * phi
    x_coords = sin_theta * np.cos(theta)
    y_coords = sin_theta * np.sin(theta)
    z_coords = cos_theta
    
    points = np.column_stack((x_coords, y_coords, z_coords))
    points = points @ R.T
    points *= radius
    return points


def _basin_hopping(initial_pos, potential, parameter, size, rng, max_iter=50, step_size=0.3):
    """
    Perform basin hopping starting from initial_pos.
    Returns (best_energy, best_positions).
    """
    best_energy = np.inf
    best_positions = initial_pos.copy()
    
    # First, relax the initial position
    try:
        energy, relaxed = relax(initial_pos, potential, parameter)
        if _check_validity(relaxed):
            best_energy = energy
            best_positions = relaxed
    except:
        pass

    for i in range(max_iter):
        if _time_remaining() < 5.0:
            break
            
        # Generate a kick
        kick = rng.normal(0, step_size, size=(size, 3))
        new_pos = best_positions + kick
        
        # Ensure the kick doesn't immediately cause overlaps (optional but helpful)
        # We rely on relax to handle it, but if relax fails on overlaps, we skip.
        
        try:
            energy, relaxed = relax(new_pos, potential, parameter)
            
            if not _check_validity(relaxed):
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            continue
            
    return best_energy, best_positions


def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses basin hopping with multiple restarts.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate radius
    radius = 1.0 * size ** (1.0 / 3.0) * 0.9
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Determine number of restarts and basin hopping iterations
    # Budget: ~45s per instance.
    # Relax is fast. Basin hopping iter is 1 relax + 1 kick.
    # Let's try 5 restarts, each with 20-30 basin hopping steps.
    n_restarts = 5
    max_bh_iter = 30
    
    for i in range(n_restarts):
        if _time_remaining() < 10.0:
            break
            
        # Generate initial configuration
        # Mix of Fibonacci sphere and random points in a sphere
        if i % 2 == 0:
            points = _fibonacci_sphere(size, radius, rng)
        else:
            # Random points in a sphere of radius radius
            points = rng.normal(0, radius/2.0, size=(size, 3))
            
        # Add small noise
        points += rng.normal(0, 0.05, size=(size, 3))
        
        # Check initial validity
        if not _check_validity(points):
            continue

        # Run basin hopping
        try:
            e_bh, pos_bh = _basin_hopping(points, potential, parameter, size, rng, max_iter=max_bh_iter)
            
            if e_bh < best_energy:
                best_energy = e_bh
                best_positions = pos_bh
        except:
            continue

    # Fallback if no valid configuration found
    if best_positions is None:
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.6
        if _time_remaining() > 2.0:
            try:
                energy, relaxed = relax(best_positions, potential, parameter)
                if _check_validity(relaxed):
                    best_positions = relaxed
            except:
                pass

    return best_positions
