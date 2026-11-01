"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import sys

def _compute_exact_radius(centers):
    """Compute the exact maximum valid radius for given centers."""
    # Distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)

def _compute_smooth_radius(centers, beta):
    """Compute a smooth approximation of the minimum radius."""
    # Wall distances
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pairwise distances
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    
    # Combine all distances
    # We want the minimum of all these distances, but smoothed.
    # Smooth min of a set of values v is -1/beta * log(sum(exp(-beta * v)))
    # To avoid underflow/overflow, we can shift by the min value.
    # However, for beta up to 1000 and distances ~0.1, exp(-100) is ~3e-44, safe in double.
    # Let's just concatenate wall_dists and the upper triangle of dists.
    
    # For efficiency, we don't need the full matrix, just unique pairs.
    # But cdist is vectorized and fast enough for N=50.
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    
    all_dists = np.concatenate([wall_dists, pair_dists])
    
    # Smooth min
    # If beta is very large, this approaches min(all_dists).
    # To improve numerical stability:
    min_d = np.min(all_dists)
    # If min_d is very small, exp(-beta * (d - min_d)) can be large for d > min_d?
    # No, exp(-beta * (d - min_d)) <= 1.
    # So we can compute: min_d - 1/beta * log(sum(exp(-beta * (all_dists - min_d))))
    shifted = all_dists - min_d
    # Clip shifted to avoid negative values if due to floating point error
    shifted = np.maximum(shifted, 0)
    exp_sum = np.sum(np.exp(-beta * shifted))
    smooth_min = min_d - (1.0 / beta) * np.log(exp_sum)
    
    # The radius is half the minimum distance between centers, or distance to wall.
    # Wait, the wall distance is the distance to the wall, which IS the radius limit for that sphere.
    # The pair distance limit is d_ij / 2.
    # So we need to be careful.
    # Let R_wall = min_i (0.5 - |c_i|)
    # Let R_pair = min_{i<j} |c_i - c_j| / 2
    # Radius r = min(R_wall, R_pair)
    
    # My current all_dists contains:
    # 1. Wall distances (which are radius limits directly)
    # 2. Pair distances (which need to be divided by 2 to be radius limits)
    
    # So I should scale the pair distances by 0.5 before combining.
    pair_dists_scaled = pair_dists * 0.5
    all_radius_limits = np.concatenate([wall_dists, pair_dists_scaled])
    
    min_lim = np.min(all_radius_limits)
    shifted = all_radius_limits - min_lim
    shifted = np.maximum(shifted, 0)
    exp_sum = np.sum(np.exp(-beta * shifted))
    smooth_min_radius = min_lim - (1.0 / beta) * np.log(exp_sum)
    
    return smooth_min_radius

def _objective(x, beta):
    """Objective function to minimize (negative smooth radius)."""
    centers = x.reshape(-1, 3)
    # Check if centers are within bounds (with small tolerance)
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 1e10
    
    r = _compute_smooth_radius(centers, beta)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = None
    best_score = -np.inf
    
    def _try_optimize(x0, beta_schedule):
        nonlocal best_x, best_score
        current_x = x0.copy()
        
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            try:
                result = minimize(_objective, current_x, args=(beta,), method="L-BFGS-B", bounds=bounds, 
                                  options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-8})
                if result.fun < -best_score:
                    # Calculate exact score to be sure
                    centers = result.x.reshape(-1, 3)
                    exact_score = _compute_exact_radius(centers)
                    if exact_score > best_score:
                        best_score = exact_score
                        best_x = result.x.copy()
                current_x = result.x.copy()
            except Exception:
                pass
        
        # Final check on the last state
        if time.time() < deadline:
            centers = current_x.reshape(-1, 3)
            exact_score = _compute_exact_radius(centers)
            if exact_score > best_score:
                best_score = exact_score
                best_x = current_x.copy()

    # Initialization 1: 5x5x2 Grid
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    centers_grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    x0_grid = centers_grid.flatten()
    
    # Beta schedule: start smooth, get sharper
    beta_schedule = [10, 50, 200, 1000]
    
    _try_optimize(x0_grid, beta_schedule)
    
    # Initialization 2: Random
    np.random.seed(42)
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        # Random points in [-0.45, 0.45]
        x0_rand = np.random.uniform(-0.45, 0.45, size=(N, 3)).flatten()
        
        # Shorter schedule for random starts
        beta_schedule_rand = [5, 20, 100]
        _try_optimize(x0_rand, beta_schedule_rand)
        
    # Final refinement on the best candidate found
    if best_x is not None and time.time() < deadline - 5:
        # Use Nelder-Mead or L-BFGS-B on the EXACT objective (non-smooth)
        # L-BFGS-B might struggle with non-smooth, but let's try a few steps.
        # Or just use a very high beta smooth version.
        try:
            result = minimize(_objective, best_x, args=(10000,), method="L-BFGS-B", bounds=bounds, 
                              options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-10})
            centers = result.x.reshape(-1, 3)
            exact_score = _compute_exact_radius(centers)
            if exact_score > best_score:
                best_score = exact_score
                best_x = result.x.copy()
        except Exception:
            pass
            
        # Also try Nelder-Mead on exact objective for local polish
        if time.time() < deadline - 2:
            def _exact_obj(x):
                centers = x.reshape(-1, 3)
                if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
                    return 1e10
                return -_compute_exact_radius(centers)
            
            try:
                result = minimize(_exact_obj, best_x, method="Nelder-Mead", bounds=bounds,
                                  options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8})
                centers = result.x.reshape(-1, 3)
                exact_score = _compute_exact_radius(centers)
                if exact_score > best_score:
                    best_score = exact_score
                    best_x = result.x.copy()
            except Exception:
                pass

    if best_x is None:
        # Fallback to grid
        best_x = x0_grid
        
    return best_x.reshape(-1, 3)

# EVOLVE-BLOCK-END
