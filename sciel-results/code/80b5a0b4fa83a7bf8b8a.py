"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def _energy(pts):
    """
    Compute the logarithmic energy: sum_{i<j} log(1 / |x_i - x_j|).
    Returns inf if any points are too close.
    """
    n = len(pts)
    # Compute pairwise distances efficiently
    # dist_sq[i, j] = ||x_i - x_j||^2
    # Using broadcasting: (pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2
    # For N=100, this is 100x100x3, which is small.
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # We only need i < j
    # Mask upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist_sq_upper = dist_sq[mask]
    
    # Check for proximity
    if np.any(dist_sq_upper < 1e-18): # 1e-9 squared
        return float('inf')
        
    dists = np.sqrt(dist_sq_upper)
    
    # Calculate sum of log(1/d) = -sum(log(d))
    # Using math.fsum for high precision as requested in task description
    # However, for speed during optimization, we can use np.sum.
    # The task says "Score: ... summed with math.fsum". 
    # The evaluator likely uses math.fsum. For our internal optimization,
    # np.sum is faster and sufficient for comparison, but to be safe 
    # and consistent, let's use math.fsum on the list of logs.
    # Actually, converting to list and using math.fsum is slower.
    # Let's use np.sum for the optimization loop, and trust that 
    # small differences don't change the ranking significantly, 
    # or use math.fsum if we have time budget.
    # Given N=100, 4950 pairs. math.fsum is not too bad.
    # Let's try np.sum first for speed in the inner loop.
    
    # log(1/d) = -log(d)
    try:
        logs = -np.log(dists)
    except (ValueError, FloatingPointError):
        return float('inf')
        
    # Check for non-finite
    if not np.all(np.isfinite(logs)):
        return float('inf')
        
    return np.sum(logs)

def _fast_energy(pts):
    """
    Faster energy calculation using vectorized operations.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Add a small epsilon to avoid log(0) if points overlap, 
    # but we check for inf separately.
    # We only sum upper triangle i < j
    # indices for upper triangle
    i, j = np.triu_indices(n, k=1)
    d_sq = dist_sq[i, j]
    
    if np.any(d_sq < 1e-18):
        return float('inf')
        
    d = np.sqrt(d_sq)
    return -np.sum(np.log(d))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    best_energy = _fast_energy(best_pts)
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-5
    alpha = 0.999 # Cooling rate per step
    
    current_pts = best_pts.copy()
    current_energy = best_energy
    T = T_start
    
    # Pre-compute indices for efficiency? No, N is small.
    
    while T > T_end:
        if time.time() > deadline:
            break
            
        # Generate a random perturbation
        # Move one point randomly
        idx = np.random.randint(N)
        
        # Random direction on sphere
        # Generate 3 normal random variables and normalize
        rand_dir = np.random.normal(size=3)
        norm = np.linalg.norm(rand_dir)
        if norm < 1e-10:
            continue
        rand_dir /= norm
        
        # Step size: adaptive or fixed? 
        # Start with a moderate step, maybe decrease as T decreases.
        step_size = 0.1 * T / T_start + 0.01
        
        # New position: current + step * dir
        new_pt = current_pts[idx] + step_size * rand_dir
        
        # Normalize to sphere
        new_pt_norm = np.linalg.norm(new_pt)
        if new_pt_norm < 1e-10:
            continue
        new_pt = new_pt / new_pt_norm
        
        # Create new configuration
        new_pts = current_pts.copy()
        new_pts[idx] = new_pt
        
        # Calculate energy change
        # To save time, we can calculate delta E instead of full E
        # E = sum log(1/d)
        # Delta E = sum_{j != idx} (log(1/d_new) - log(1/d_old))
        # d_old = ||current_pts[idx] - current_pts[j]||
        # d_new = ||new_pt - current_pts[j]||
        
        old_dists_sq = np.sum((current_pts[idx] - current_pts) ** 2, axis=1)
        new_dists_sq = np.sum((new_pt - current_pts) ** 2, axis=1)
        
        # Exclude self
        old_dists_sq[idx] = 1.0 # Placeholder, will be masked
        new_dists_sq[idx] = 1.0
        
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        
        old_dists = np.sqrt(old_dists_sq[mask])
        new_dists = np.sqrt(new_dists_sq[mask])
        
        # Check for proximity
        if np.any(old_dists < 1e-9) or np.any(new_dists < 1e-9):
            # If old was valid and new is not, reject
            # If new is not valid, delta is +inf
            if np.any(new_dists < 1e-9):
                delta_E = float('inf')
            else:
                # Old was invalid? We assume current state is always valid in SA.
                # If current state becomes invalid, we should revert.
                delta_E = float('inf')
        else:
            delta_E = np.sum(np.log(old_dists)) - np.sum(np.log(new_dists))
            # log(1/d_new) - log(1/d_old) = -log(d_new) - (-log(d_old)) = log(d_old) - log(d_new)
            
        # Acceptance probability
        if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
            current_pts = new_pts
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        
        T *= alpha
        
    # Final verification with precise math.fsum energy
    # The task requires the score to be calculated with math.fsum.
    # Our internal tracking used np.sum. We should return the points.
    # The evaluator will recalculate the score.
    # We just need to ensure the points are valid.
    
    # Normalize best_pts to be safe
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
