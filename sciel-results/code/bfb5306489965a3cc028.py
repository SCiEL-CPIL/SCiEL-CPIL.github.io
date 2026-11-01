"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
import random


def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    dists = cdist(c, c)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)


def _initial_configuration():
    """Create a good initial configuration using a BCC-like arrangement."""
    # Start with a 4x4x4 grid (64 points) and pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
    
    # Add some BCC-like points to improve packing
    # Shift some points slightly to create BCC structure
    random.seed(42)
    random.shuffle(points)
    return np.array(points[:N])


def _local_refine(c, max_iters=200, tol=1e-10):
    """Local refinement using gradient-free optimization."""
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            for dim in range(3):
                # Try small perturbations
                for delta in [1e-4, 5e-5, 1e-5, -1e-4, -5e-5, -1e-5]:
                    c_try = best_c.copy()
                    c_try[i, dim] += delta
                    # Check bounds
                    if c_try[i, dim] < -0.5 or c_try[i, dim] > 0.5:
                        continue
                    r_try = _compute_radius(c_try)
                    if r_try > best_r + tol:
                        best_c = c_try
                        best_r = r_try
                        improved = True
                        break
        if not improved:
            break
    
    return best_c, best_r


def _simulated_annealing(c, time_budget=100.0):
    """Simulated annealing to optimize the packing."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    T = 0.01  # Initial temperature
    cooling = 0.999
    min_T = 1e-8
    
    c_current = c.copy()
    r_current = _compute_radius(c_current)
    
    while time.time() - start_time < time_budget:
        if T < min_T:
            break
        
        # Random perturbation
        c_try = c_current.copy()
        i = random.randint(0, N - 1)
        dim = random.randint(0, 2)
        delta = random.gauss(0, T * 0.5)
        c_try[i, dim] += delta
        
        # Check bounds
        if c_try[i, dim] < -0.5 or c_try[i, dim] > 0.5:
            T *= cooling
            continue
        
        r_try = _compute_radius(c_try)
        
        # Acceptance criterion
        if r_try > r_current or random.random() < np.exp((r_try - r_current) / T):
            c_current = c_try
            r_current = r_try
            if r_current > best_r:
                best_c = c_current.copy()
                best_r = r_current
        
        T *= cooling
    
    return best_c, best_r


def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration
    c = _initial_configuration()
    
    # Local refinement
    c, best_r = _local_refine(c, max_iters=100)
    
    # Simulated annealing if time permits
    if time.time() < deadline - 20.0:
        c_sa, r_sa = _simulated_annealing(c, time_budget=min(80.0, deadline - time.time() - 10.0))
        if r_sa > best_r:
            c = c_sa
            best_r = r_sa
    
    # Final local refinement
    if time.time() < deadline - 5.0:
        c, best_r = _local_refine(c, max_iters=50)
    
    # Ensure validity
    c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c
# EVOLVE-BLOCK-END
