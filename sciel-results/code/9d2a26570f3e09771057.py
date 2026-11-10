"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Boundary constraint: min distance from center to any edge
    # Square is [-0.5, 0.5]^2
    margins = np.minimum(0.5 - np.abs(centers), axis=1)
    r_boundary = np.min(margins)
    
    # Pairwise distances
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair_dist:
                min_pair_dist = d
    r_pair = min_pair_dist / 2.0
    
    return min(r_boundary, r_pair)

def _negative_radius(centers_flat):
    """Objective function: negative of the radius (to minimize)."""
    centers = centers_flat.reshape(-1, 2)
    r = _compute_radius(centers)
    return -r

def _smooth_negative_radius(centers_flat, sharpness=50.0):
    """Smooth approximation using log-sum-exp for the min operation."""
    centers = centers_flat.reshape(-1, 2)
    n = len(centers)
    
    # Boundary margins
    margins = 0.5 - np.abs(centers)
    
    # Pairwise distances / 2
    pair_dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j]) / 2.0
            pair_dists[i, j] = d
            pair_dists[j, i] = d
    
    # Collect all constraints
    all_constraints = np.concatenate([margins.flatten(), pair_dists[pair_dists > 0].flatten()])
    
    # Log-sum-exp approximation of min: min(x) ≈ -1/k * log(sum(exp(-k*x)))
    # This is a smooth lower bound that approaches min as k→∞
    k = sharpness
    shifted = -k * all_constraints
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    smooth_min = -log_sum_exp / k
    
    return -smooth_min

def _initial_grid():
    """Generate an initial grid layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _initial_random(seed):
    """Generate a random initial layout."""
    rng = np.random.RandomState(seed)
    return (rng.rand(N, 2) - 0.5) * 0.9

def construct():
    """Construct centers of N equal circles in [-0.5, 0.5]^2 maximizing radius."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    best_centers = _initial_grid()
    best_radius = _compute_radius(best_centers)
    
    # Multiple restarts with different initializations and optimization methods
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Choose initialization
        if attempts % 3 == 1:
            init = _initial_grid()
        elif attempts % 3 == 2:
            init = _initial_random(seed=attempts * 137)
        else:
            # Perturbed best
            rng = np.random.RandomState(attempts * 251)
            init = best_centers + rng.randn(N, 2) * 0.02
            # Clamp to valid region
            init = np.clip(init, -0.49, 0.49)
        
        init_flat = init.flatten()
        
        # Try Nelder-Mead on the exact non-smooth objective
        try:
            res = minimize(
                _negative_radius,
                init_flat,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res.success or res.fun < -best_radius:
                r = -res.fun
                if r > best_radius:
                    best_radius = r
                    best_centers = res.x.reshape(-1, 2)
                    best_centers = np.clip(best_centers, -0.4999, 0.4999)
        except Exception:
            pass
        
        # If time permits, also try smooth version with L-BFGS-B
        if time.time() < deadline - 10.0:
            try:
                res2 = minimize(
                    _smooth_negative_radius,
                    init_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'gtol': 1e-8}
                )
                # Evaluate actual radius at the smooth solution
                centers2 = res2.x.reshape(-1, 2)
                centers2 = np.clip(centers2, -0.4999, 0.4999)
                r2 = _compute_radius(centers2)
                if r2 > best_radius:
                    best_radius = r2
                    best_centers = centers2
            except Exception:
                pass
        
        # If time permits, do a local refinement with Nelder-Mead from current best
        if time.time() < deadline - 5.0 and attempts >= 2:
            try:
                rng = np.random.RandomState(attempts * 997)
                perturbed = best_centers + rng.randn(N, 2) * 0.005
                perturbed = np.clip(perturbed, -0.4999, 0.4999)
                res3 = minimize(
                    _negative_radius,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                r3 = -res3.fun
                if r3 > best_radius:
                    best_radius = r3
                    best_centers = res3.x.reshape(-1, 2)
                    best_centers = np.clip(best_centers, -0.4999, 0.4999)
            except Exception:
                pass
    
    # Final validation: ensure all centers are strictly inside
    best_centers = np.clip(best_centers, -0.499999, 0.499999)
    
    return best_centers
# EVOLVE-BLOCK-END
