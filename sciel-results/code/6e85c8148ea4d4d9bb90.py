"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _compute_score(centers):
    """
    Compute the score for a given set of centers.
    Score = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    Returns -1 if invalid (centers outside cube, non-finite, etc.)
    """
    if not np.all(np.isfinite(centers)):
        return -1.0
    
    # Check if centers are within the cube [-0.5, 0.5]^3
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return -1.0
    
    # Calculate pairwise distances
    sq_dists = np.sum(centers**2, axis=1)
    dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * centers @ centers.T
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    max_abs_coord = np.max(np.abs(centers))
    min_wall_dist = 0.5 - max_abs_coord
    
    score = min(min_pair_dist / 2.0, min_wall_dist)
    
    # Check validity: centers must be strictly inside
    if np.any(centers <= -0.5) or np.any(centers >= 0.5):
        return -1.0
    
    return score

def _smooth_objective(centers_flat, beta=50.0):
    """
    Smooth approximation of the negative minimum clearance.
    Uses log-sum-exp to approximate the min distance.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # Pairwise distances
    sq_dists = np.sum(centers**2, axis=1)
    dists_sq = sq_dists[:, np.newaxis] + sq_dists[np.newaxis, :] - 2 * centers @ centers.T
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(np.maximum(dists_sq, 0))
    
    # Wall distances
    wall_dists = np.abs(centers)
    # We want min(0.5 - |c_ik|). This is equivalent to 0.5 - max(|c_ik|).
    # For log-sum-exp, we need to sum exp(-beta * d).
    # d_pair = dists / 2.0
    # d_wall = 0.5 - np.abs(centers)
    
    d_pairs = dists / 2.0
    d_walls = 0.5 - np.abs(centers)
    
    # Flatten for sum
    d_pairs_flat = d_pairs[np.triu_indices(N, k=1)]
    d_walls_flat = d_walls.flatten()
    
    all_dists = np.concatenate([d_pairs_flat, d_walls_flat])
    
    # Handle potential negatives (shouldn't happen if we clip, but safe)
    all_dists = np.maximum(all_dists, 1e-9)
    
    # Log-sum-exp approximation of min
    # min(x) approx -1/beta * log(sum(exp(-beta * x)))
    # We want to minimize -min(x), so we maximize min(x).
    # Objective = -min(x) approx 1/beta * log(sum(exp(-beta * x)))
    
    # To avoid overflow/underflow, subtract max
    max_d = np.max(all_dists)
    log_sum = np.log(np.sum(np.exp(-beta * (all_dists - max_d))))
    smooth_min = max_d - log_sum / beta
    
    # We want to MINIMIZE the negative of the score.
    # Score = smooth_min.
    # So objective = -smooth_min.
    return -smooth_min

def construct():
    # 1. Initial Guess: Simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_centers = grid[:N]
    
    # Ensure initial centers are strictly inside
    initial_centers = np.clip(initial_centers, -0.49, 0.49)
    
    best_centers = initial_centers.copy()
    best_score = _compute_score(best_centers)
    
    # 2. Optimization using L-BFGS-B on the smooth objective
    # We use multiple betas to refine the solution.
    betas = [10.0, 50.0, 200.0]
    
    current_centers = initial_centers.copy()
    
    for beta in betas:
        # Flatten for optimizer
        x0 = current_centers.flatten()
        
        # Bounds: strictly inside [-0.5, 0.5]
        bounds = [(-0.4999, 0.4999)] * (N * 3)
        
        try:
            res = minimize(
                _smooth_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 1000, 'ftol': 1e-10, 'beta': beta}
            )
            new_centers = res.x.reshape(N, 3)
            
            # Check if valid
            if np.all(np.isfinite(new_centers)):
                # Clip to ensure strict validity
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_score = _compute_score(new_centers)
                
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers
                    current_centers = new_centers
        except Exception:
            pass
            
    # 3. Local refinement with exact score (hill climbing)
    def _local_refine(centers, max_iters=500, step=0.01):
        current_score = _compute_score(centers)
        if current_score < 0:
            return centers, 0.0
        
        for _ in range(max_iters):
            improved = False
            for _ in range(20):
                perturbed = centers + np.random.normal(0, step, size=centers.shape)
                perturbed = np.clip(perturbed, -0.4999, 0.4999)
                new_score = _compute_score(perturbed)
                if new_score > current_score:
                    centers = perturbed
                    current_score = new_score
                    improved = True
            if not improved:
                step *= 0.9
                if step < 1e-7:
                    break
        return centers, current_score
    
    best_centers, best_score = _local_refine(best_centers, max_iters=1000, step=0.02)
    
    # Final safety clip
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers

# EVOLVE-BLOCK-END
