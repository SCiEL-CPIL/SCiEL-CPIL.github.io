"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _soft_min(x, t=0.01):
    """Soft minimum via log-sum-exp for differentiable min."""
    # For soft min: -softmax(-x/t) * t
    # To avoid overflow, subtract max
    x_min = np.min(x)
    # If all are very negative, this is fine
    # exp((x - x_min)/t)
    exp_x = np.exp((x - x_min) / t)
    sum_exp = np.sum(exp_x, axis=-1)
    log_sum_exp = np.log(sum_exp)
    return x_min - t * log_sum_exp

def _clearance_sum_smooth(pos, t=0.01):
    """
    Objective function for optimization.
    Minimizes negative sum of clearances using soft-min for differentiability.
    pos: shape (2*N,)
    """
    p = pos.reshape(N, 2)
    # Calculate pairwise distances squared first for efficiency/stability? No, need sqrt for distances.
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    dist = np.sqrt(dist_sq + 1e-12)
    
    # Mask diagonal with large value so soft-min ignores it
    mask = ~np.eye(N, dtype=bool)
    dist_masked = np.where(mask, dist, 1e6)
    
    # Minimum distance to any other point (soft)
    min_dist_to_others = -_soft_min(-dist_masked, t) # soft min
    
    # Distance to box boundaries
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    # Clearance for each point
    # r_i <= 0.5 * dist(i, j) ensures r_i + r_j <= dist(i, j)
    # We want to maximize sum of r_i. 
    # r_i is limited by distance to box and 0.5 * min_dist_to_others.
    # So r_i = min(dist_to_box, 0.5 * min_dist_to_others)
    # We want to maximize sum(r_i).
    # Let's use soft-min for this too to keep it smooth?
    # Or just use the hard min for the final value, but the objective for optimization can be the soft version.
    # Actually, maximizing the sum of the hard min is non-smooth.
    # Let's define the objective as minimizing -sum(soft_min(dist_to_box, 0.5*min_dist_to_others))
    
    limits = np.stack([dist_to_box, 0.5 * min_dist_to_others], axis=1)
    soft_clearances = -_soft_min(-limits, t)
    
    return -np.sum(soft_clearances)

def _get_valid_packing(pos):
    """
    Convert optimized positions to a valid packing with radii.
    Returns (positions (N,2), radii (N,))
    """
    p = pos.reshape(N, 2).copy()
    # Clamp to unit square
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    
    # Calculate radii based on clearances (hard min)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    min_dist_to_others = np.min(dist, axis=1)
    dist_to_box = np.min(np.stack([p[:, 0], 1.0 - p[:, 0], p[:, 1], 1.0 - p[:, 1]], axis=1), axis=1)
    
    radii = np.minimum(dist_to_box, 0.5 * min_dist_to_others)
    
    # Apply safety factor to ensure strict non-overlap
    safety = 0.9999
    radii *= safety
    
    return p, radii

def _hex_initialization():
    """Generate a hexagonal grid initialization."""
    pts = []
    # Use a grid density that provides at least N points
    # 6x6 = 36 points
    for i in range(6):
        for j in range(6):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 6.0
            # Stagger rows
            if i % 2 == 0:
                y += 0.5 / 6.0 * 0.8
            # Ensure within bounds
            if 0.0 < x < 1.0 and 0.0 < y < 1.0:
                pts.append([x, y])
    
    if len(pts) < N:
        while len(pts) < N:
            pts.append([np.random.rand(), np.random.rand()])
            
    return np.array(pts[:N]).flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # List of initializations
    inits = [
        _hex_initialization(),
        np.random.rand(2 * N),
        np.random.rand(2 * N),
        np.random.rand(2 * N),
        np.random.rand(2 * N)
    ]
    
    for init_pos in inits:
        if time.time() > deadline - 10.0:
            break
            
        try:
            # Use L-BFGS-B for local optimization on the smooth objective
            res = minimize(_clearance_sum_smooth, init_pos, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Evaluate the result with the hard min (actual score)
            pos, radii = _get_valid_packing(res.x)
            score = np.sum(radii)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack((pos, radii))
                
        except Exception:
            continue
            
    # Fallback if optimization failed
    if best_result is None:
        pos, radii = _get_valid_packing(inits[0])
        best_result = np.column_stack((pos, radii))
        
    return best_result

# EVOLVE-BLOCK-END
