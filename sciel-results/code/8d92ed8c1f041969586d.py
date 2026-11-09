"""Morse cluster global minima search with structured initialization and deadline guard."""
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


def _fibonacci_sphere(n, radius, rng):
    """
    Generate n points on a sphere of given radius using Fibonacci lattice.
    This provides a uniform distribution which is good for initial packing.
    """
    # Golden angle step
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n)
    i = i + 0.5
    cos_theta = 1.0 - (i / n)
    sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
    
    # Random rotation to avoid alignment with axes
    # Generate a random rotation matrix
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])
    
    # Coordinates on unit sphere
    theta = i * phi
    x_coords = sin_theta * np.cos(theta)
    y_coords = sin_theta * np.sin(theta)
    z_coords = cos_theta
    
    points = np.column_stack((x_coords, y_coords, z_coords))
    # Apply rotation
    points = points @ R.T
    # Scale to radius
    points *= radius
    return points


def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses multiple restarts with structured (Fibonacci sphere) initialization.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Estimate radius: equilibrium distance is 1.0. 
    # For a cluster of n atoms, radius scales as n^(1/3).
    # A factor of 0.8-1.0 is typical for compact clusters.
    radius = 1.0 * size ** (1.0 / 3.0) * 0.9
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility in testing, but we vary rotation
    
    best_energy = np.inf
    best_positions = None
    
    # Determine number of restarts based on time and size
    # Larger n is slower to relax. 
    # Let's aim for ~10-20 restarts for small n, fewer for large n if time is tight.
    # Relaxation is usually fast (<1s). 
    # We have ~45s per instance. 20 restarts * 1s = 20s. Safe.
    max_restarts = 20
    
    for i in range(max_restarts):
        if _time_remaining() < 5.0:
            break
            
        # Generate initial configuration
        points = _fibonacci_sphere(size, radius, rng)
        
        # Add small random noise to break symmetry and avoid saddle points
        noise = rng.normal(0, 0.05, size=(size, 3))
        points += noise
        
        # Check initial min distance
        # If min dist < 0.5, this start is invalid or risky. 
        # With radius ~ n^(1/3), dist should be > 1.0 usually.
        # But let's be safe.
        if size > 1:
            # Efficient min dist check for initial validation
            # For n=80, cdist is 80x80, very fast.
            dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=-1)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 0.5:
                continue

        try:
            # Relax the configuration
            energy, relaxed_positions = relax(points, potential, parameter)
            
            # Validate result
            if relaxed_positions is None or len(relaxed_positions) != size:
                continue
                
            # Check final min distance
            dists = np.linalg.norm(relaxed_positions[:, np.newaxis, :] - relaxed_positions[np.newaxis, :, :], axis=-1)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.5:
                # Invalid configuration according to task constraints
                continue
                
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_positions
                
        except Exception as e:
            # If relax fails, skip this restart
            continue
            
    # If no valid configuration was found (should not happen with Fib sphere), 
    # return a fallback valid configuration
    if best_positions is None:
        # Fallback: Linear chain or simple grid, ensuring min dist >= 0.5
        # Simple linear chain along x-axis
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.6 # Spacing 0.6 > 0.5
        # Calculate energy for this fallback just in case
        # But we should try to relax this fallback too if time permits
        if _time_remaining() > 2.0:
            try:
                energy, relaxed_positions = relax(best_positions, potential, parameter)
                dists = np.linalg.norm(relaxed_positions[:, np.newaxis, :] - relaxed_positions[np.newaxis, :, :], axis=-1)
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) >= 0.5:
                    best_positions = relaxed_positions
            except:
                pass

    return best_positions
