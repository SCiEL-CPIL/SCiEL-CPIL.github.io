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

def _force_directed_relaxation(c, max_iters=1000, alpha_start=0.05, alpha_end=0.001, beta=2.0):
    """
    Deterministic Force-Directed Relaxation.
    Spheres repel each other with force ~ 1/d^beta.
    Walls repel spheres when they get close.
    """
    c = c.copy()
    best_r = _compute_radius(c)
    
    # Pre-compute indices for efficiency
    indices = np.arange(N)
    
    for it in range(max_iters):
        # Anneal the step size (alpha)
        alpha = alpha_start * (alpha_end / alpha_start) ** (it / max_iters)
        
        # Calculate forces
        forces = np.zeros_like(c)
        
        # Pairwise repulsion
        dists = cdist(c, c)
        np.fill_diagonal(dists, np.inf)
        
        # Avoid division by zero or very small numbers
        dists_safe = np.maximum(dists, 1e-6)
        
        # Force magnitude: k / d^beta
        # Direction: (c_j - c_i) / d
        # We want to push i away from j, so force on i is opposite to (c_j - c_i)
        # F_ij = (c_i - c_j) / d^2 * (1/d^beta) = (c_i - c_j) / d^(beta+2)
        
        # Vectorized force calculation
        # dists shape (N, N)
        # diff shape (N, N, 3)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        
        # Apply force only if dist < threshold (optional, but here we use all)
        # To avoid instability, we cap the force magnitude
        inv_d_pow = 1.0 / (dists_safe ** (beta + 2))
        
        # Zero out diagonal
        np.fill_diagonal(inv_d_pow, 0.0)
        
        # Force on i from j is diff[i,j] * inv_d_pow[i,j]
        # Sum over j
        forces += np.sum(diff * inv_d_pow[:, :, np.newaxis], axis=1)
        
        # Wall forces
        # If c < -0.5, force should be positive (push right)
        # If c > 0.5, force should be negative (push left)
        # We use a soft potential: if dist_to_wall < margin, force = k_wall * (margin - dist)
        margin = 0.1
        k_wall = 5.0
        
        # Left wall (x < -0.5)
        dist_left = c + 0.5
        mask_left = dist_left < margin
        forces[mask_left] += k_wall * (margin - dist_left[mask_left])[:, np.newaxis]
        
        # Right wall (x > 0.5)
        dist_right = 0.5 - c
        mask_right = dist_right < margin
        forces[mask_right] -= k_wall * (margin - dist_right[mask_right])[:, np.newaxis]
        
        # Apply update
        c += alpha * forces
        
        # Clamp to cube
        c = np.clip(c, -0.5, 0.5)
        
        # Check convergence
        if it % 10 == 0:
            r_try = _compute_radius(c)
            if r_try > best_r:
                best_r = r_try
            if np.max(np.abs(alpha * forces)) < 1e-8:
                break
                
    return c, best_r

def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration
    c = _initial_configuration()
    
    # 1. Force Directed Relaxation
    c, r_fdr = _force_directed_relaxation(c, max_iters=500)
    
    # 2. Local Refinement
    c, best_r = _local_refine(c, max_iters=100)
    
    # 3. Simulated Annealing (if time permits, as a secondary polish or escape mechanism)
    # We use SA here to potentially escape local minima found by FDR
    if time.time() < deadline - 30.0:
        # Use a shorter SA run
        budget = min(50.0, deadline - time.time() - 10.0)
        if budget > 5.0:
            c_sa, r_sa = _simulated_annealing(c, time_budget=budget)
            if r_sa > best_r:
                c = c_sa
                best_r = r_sa
    
    # 4. Final Local Refinement
    if time.time() < deadline - 5.0:
        c, best_r = _local_refine(c, max_iters=100)
        
    # Ensure validity
    c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c
# EVOLVE-BLOCK-END
