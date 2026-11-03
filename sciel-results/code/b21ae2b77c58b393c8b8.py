"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

# Define ring structure: 1 center + 5 rings
# Sum: 1 + 6 + 12 + 12 + 12 + 7 = 50
RING_COUNTS = [1, 6, 12, 12, 12, 7]
NUM_RINGS = len(RING_COUNTS) - 1 # Exclude center

def get_centers_from_params(params):
    """
    params: array of length 2 * NUM_RINGS
    [r1, r2, ..., rN, a1, a2, ..., aN]
    """
    centers = np.zeros((N, 2))
    idx = 0
    # Center
    centers[idx] = [0.0, 0.0]
    idx += 1
    
    radii = params[:NUM_RINGS]
    angles = params[NUM_RINGS:]
    
    for i, count in enumerate(RING_COUNTS[1:], 1):
        r = radii[i-1]
        a_offset = angles[i-1]
        for j in range(count):
            theta = a_offset + 2 * np.pi * j / count
            centers[idx, 0] = r * np.cos(theta)
            centers[idx, 1] = r * np.sin(theta)
            idx += 1
    return centers

def objective_low_dim(params):
    centers = get_centers_from_params(params)
    # Boundary distance
    dist_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return -min(min_boundary, min_pairwise)

def objective_full(centers_flat):
    centers = centers_flat.reshape(-1, 2)
    dist_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    min_boundary = np.min(dist_boundary)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return -min(min_boundary, min_pairwise)

def construct():
    start_time = time.time()
    # Target to finish well before 120s, say 100s
    deadline = start_time + 100.0
    
    # 1. Global search in low-dimensional space
    bounds = [(0.1, 1.0)] * NUM_RINGS + [(0, 2*np.pi)] * NUM_RINGS
    
    # Initial guess for DE
    # Radii should be increasing
    init_radii = np.linspace(0.2, 0.9, NUM_RINGS)
    init_angles = np.zeros(NUM_RINGS)
    init_params = np.concatenate([init_radii, init_angles])
    
    try:
        # DE is stochastic. Keep popsize small to fit in time.
        res = differential_evolution(
            objective_low_dim, 
            bounds, 
            x0=init_params,
            maxiter=100, 
            popsize=10, 
            tol=0.01, 
            seed=42,
            polish=False # We will polish later
        )
        
        if res.success:
            best_params = res.x
        else:
            best_params = init_params
            
    except Exception:
        best_params = init_params
        
    # 2. Expand to full coordinates
    best_centers = get_centers_from_params(best_params)
    
    # 3. Refine with L-BFGS-B
    if time.time() < deadline - 20: # Leave 20s buffer
        try:
            x0 = best_centers.flatten()
            res_full = minimize(
                objective_full, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_full.success:
                candidate = res_full.x.reshape(-1, 2)
                # Validate
                if np.all(np.isfinite(candidate)):
                    best_centers = candidate
        except Exception:
            pass
            
    return best_centers
# EVOLVE-BLOCK-END
