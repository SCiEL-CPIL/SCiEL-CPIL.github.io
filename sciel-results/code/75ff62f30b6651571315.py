"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import pdist

def get_score(pts):
    """Calculate the squared ratio of max to min pairwise distance."""
    dists = pdist(pts)
    if np.any(dists < 1e-10):
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d == 0:
        return 1e10
    return (max_d / min_d) ** 2

def objective(x):
    """Flatten points for scipy optimizers."""
    pts = x.reshape(N, D)
    return get_score(pts)

def construct():
    # Define bounds for DE. 
    # The problem is scale-invariant, so we bound to [-5, 5]^3.
    bounds = [(-5, 5) for _ in range(N * D)]
    
    # Differential Evolution settings
    # popsize: 20 * 42 = 840 individuals
    # maxiter: 300
    # This provides a robust global search.
    de_result = differential_evolution(
        objective, 
        bounds, 
        maxiter=300, 
        popsize=20, 
        tol=1e-8, 
        seed=42,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True
    )
    
    best_pts = de_result.x.reshape(N, D)
    best_score = de_result.fun
    
    # Additional local refinement with Powell's method for high precision
    # Powell is robust for non-smooth functions (min/max switch)
    local_result = minimize(
        objective, 
        best_pts.flatten(), 
        method='Powell', 
        options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12}
    )
    
    if local_result.fun < best_score:
        best_pts = local_result.x.reshape(N, D)
        best_score = local_result.fun
        
    return best_pts

# EVOLVE-BLOCK-END
