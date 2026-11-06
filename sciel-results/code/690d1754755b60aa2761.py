import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    # Reuse seed logic but return array
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= n:
            # Sort by radius to keep center points? Or just take first N?
            # Taking first N might bias towards corners if not sorted.
            # Seed sorts by radius. We do the same.
            return points[np.argsort(radius, kind="stable")[:n]]
        spacing *= 0.98

def _compute_radius(points):
    # points shape (N, 2)
    # Return min distance / 2 and min boundary dist
    dists = np.sqrt(np.sum(points**2, axis=1))
    boundary_r = np.min(1.0 - dists)
    
    # Pairwise distances
    # Using broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    
    return min(boundary_r, pair_r)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # 1. Initialization
    P = _hex_lattice_init(N)
    current_r = _compute_radius(P)
    best_P = P.copy()
    best_r = current_r
    
    # 2. Optimization with Soft-Min Annealing
    betas = [5, 20, 50, 100, 200, 500, 1000, 2000]
    
    def objective_flat(flat_P, beta):
        P = flat_P.reshape(-1, 2)
        # Compute all relevant distances
        dists = np.sqrt(np.sum(P**2, axis=1))
        boundary_dists = 1.0 - dists
        
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        pair_dists = dist_matrix / 2.0
        
        # Combine into one vector
        # Exclude diagonal for pairs
        pair_vals = pair_dists[np.triu_indices(N, k=1)]
        all_vals = np.concatenate([pair_vals, boundary_dists])
        
        # Soft min
        min_val = np.min(all_vals)
        # Shift for stability
        shifted = all_vals - min_val
        # Sum of exps
        # Clip to avoid overflow in exp
        log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min = min_val - (1.0 / beta) * log_sum_exp
        
        return -soft_min

    P_flat = P.flatten()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        # Define objective with current beta
        def obj(x):
            return objective_flat(x, beta)
            
        try:
            res = minimize(obj, P_flat, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.success or res.fun < objective_flat(P_flat, beta):
                P_flat = res.x
                P = P_flat.reshape(-1, 2)
                r = _compute_radius(P)
                if r > best_r:
                    best_r = r
                    best_P = P.copy()
        except Exception:
            pass
            
    # Final check
    return best_P
