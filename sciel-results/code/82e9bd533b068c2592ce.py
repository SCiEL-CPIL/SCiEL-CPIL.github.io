"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def compute_radius(centers):
    """
    Compute the maximum radius for a given set of centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_{i,k} (0.5 - |c_ik|) )
    """
    if not np.all(np.isfinite(centers)):
        return -1.0
        
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq))
    
    abs_centers = np.abs(centers)
    dists_to_boundary = 0.5 - abs_centers
    min_boundary_dist = np.min(dists_to_boundary)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def objective(x):
    centers = x.reshape((N, 3))
    centers = np.clip(centers, -0.5, 0.5)
    return -compute_radius(centers)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    # Differential Evolution
    # popsize=15 is the minimum recommended. 
    # 15 * 150 = 2250 individuals.
    # Let's try maxiter=100.
    # If it's too slow, we can reduce maxiter.
    
    res = differential_evolution(
        objective, 
        bounds, 
        maxiter=100, 
        popsize=15, 
        seed=42,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True  # This uses L-BFGS-B by default, which might fail on non-smooth.
                     # We will do our own polish. Set polish=False to avoid overhead/failures.
    )
    
    # Actually, setting polish=True in DE uses L-BFGS-B. 
    # Our function is non-smooth (min). L-BFGS-B assumes smoothness. 
    # It might get stuck or behave poorly. 
    # Better to set polish=False and do Nelder-Mead ourselves.
    
    res = differential_evolution(
        objective, 
        bounds, 
        maxiter=100, 
        popsize=15, 
        seed=42,
        tol=1e-8,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False
    )
    
    best_centers = res.x.reshape((N, 3))
    
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        # Nelder-Mead for local refinement
        res2 = minimize(
            objective, 
            best_centers, 
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-12}
        )
        if res2.fun < res.fun:
            best_centers = res2.x.reshape((N, 3))
            
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers
# EVOLVE-BLOCK-END
