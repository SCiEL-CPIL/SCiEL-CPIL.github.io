import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """
    Computes the maximum radius for a given set of centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    # Pairwise distances
    dists = cdist(centers, centers)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    # Distance to each of the 4 boundaries: 0.5 - x, 0.5 + x, 0.5 - y, 0.5 + y
    # Which is min(0.5 - |x|, 0.5 - |y|)
    boundary_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    min_boundary_dist = np.min(boundary_dists)
    
    return min(min_pair_dist, 2 * min_boundary_dist) / 2.0

def _smooth_objective(flat_centers, beta=20.0):
    """
    Smooth approximation of the negative radius.
    We minimize this, so we maximize the radius.
    """
    centers = flat_centers.reshape(N, 2)
    
    # Pairwise distances
    # Compute squared distances to avoid sqrt in exp? 
    # d = sqrt(sum((x_i - x_j)^2))
    # exp(-beta * d)
    # For speed, we can compute all pairwise diffs
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists = dists[mask]
    
    # Boundary distances
    boundary_dists = np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1]))
    
    # Combine all constraints
    all_dists = np.concatenate([dists, boundary_dists])
    
    # Smooth min: -1/beta * log(sum(exp(-beta * d)))
    # To avoid overflow/underflow in exp, shift by max
    # Actually, if d is small, exp(-beta*d) is large.
    # We want to penalize small d.
    # min(d) is approximated by -1/beta * log(sum(exp(-beta d)))
    # If one d is very small, exp(-beta d) is large, log is large, -1/beta * log is negative large.
    # Wait.
    # Let d_min = 0.1. d_other = 0.2.
    # sum = exp(-2) + exp(-4) = 0.135 + 0.018 = 0.153.
    # log(0.153) = -1.87.
    # -1/20 * (-1.87) = 0.0935.
    # True min is 0.1.
    # As beta -> inf, it approaches 0.1.
    # We want to MINIMIZE -radius.
    # So we minimize - (smooth_min).
    # smooth_min is positive.
    # So we return -smooth_min.
    
    max_d = np.max(all_dists)
    # Shift to prevent underflow for large d? 
    # exp(-beta * d). If d is large, exp is small.
    # If d is small, exp is large.
    # We can just compute it directly if beta is not too huge and d is not too small.
    # d is around 0.1. beta=20. exp(-2) is fine.
    
    vals = np.exp(-beta * all_dists)
    sum_exp = np.sum(vals)
    smooth_min = -np.log(sum_exp) / beta
    
    return -smooth_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Initializations
    # 1. 5x10 Grid
    m = 5
    n = 10
    x = (np.arange(m) + 0.5) / m - 0.5
    y = (np.arange(n) + 0.5) / n - 0.5
    grid_5_10 = np.array([(xi, yj) for yj in y for xi in x])
    
    # 2. 10x5 Grid
    x2 = (np.arange(n) + 0.5) / n - 0.5
    y2 = (np.arange(m) + 0.5) / m - 0.5
    grid_10_5 = np.array([(xi, yj) for yj in y2 for xi in x2])
    
    # 3. 7x7 Grid + Center
    m3 = 7
    x3 = (np.arange(m3) + 0.5) / m3 - 0.5
    y3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid_7_7 = np.array([(xi, yj) for yj in y3 for xi in x3])
    center_pt = np.array([[0.0, 0.0]])
    grid_7_7_plus = np.vstack([grid_7_7, center_pt])
    
    # 4. Random
    np.random.seed(42)
    rand_centers = np.random.uniform(-0.4, 0.4, size=(N, 2))
    
    initializations = [grid_5_10, grid_10_5, grid_7_7_plus, rand_centers]
    
    # Add some random perturbations of the grids
    for _ in range(10):
        perturbed = grid_5_10 + np.random.uniform(-0.05, 0.05, size=(N, 2))
        initializations.append(perturbed)
        
    for _ in range(10):
        perturbed = grid_10_5 + np.random.uniform(-0.05, 0.05, size=(N, 2))
        initializations.append(perturbed)

    for init_centers in initializations:
        if time.time() > deadline:
            break
            
        # Ensure valid start (clip to bounds)
        init_centers = np.clip(init_centers, -0.5, 0.5)
        
        # Check current radius
        current_r = _compute_radius(init_centers)
        if current_r > best_radius:
            best_radius = current_r
            best_centers = init_centers.copy()
            
        # Optimize
        try:
            res = minimize(
                _smooth_objective, 
                init_centers.flatten(), 
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_centers = res.x.reshape(N, 2)
            opt_centers = np.clip(opt_centers, -0.5, 0.5)
            
            opt_r = _compute_radius(opt_centers)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers.copy()
                
        except Exception:
            continue
            
    # Final check
    if best_centers is None:
        return grid_5_10
        
    return best_centers
# EVOLVE-BLOCK-END
