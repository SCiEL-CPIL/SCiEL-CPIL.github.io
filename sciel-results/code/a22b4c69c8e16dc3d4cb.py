"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    # Pairwise distances
    d = np.linalg.norm(centres[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair_dist = np.min(d)
    
    # Distance to walls
    wall_dist = 0.5 - np.abs(centres)
    min_wall_dist = np.min(wall_dist)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _neg_radius(centres):
    return -_compute_radius(centres)


def _local_optimize(centres, time_limit=110.0):
    """Local optimization using gradient ascent on minimum distance."""
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Use Nelder-Mead or L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try multiple optimization methods and keep the best
    methods = ['Nelder-Mead', 'L-BFGS-B']
    
    for method in methods:
        if time.time() - start_time > time_limit:
            break
            
        try:
            result = minimize(
                _neg_radius,
                best_centres.flatten(),
                method=method,
                bounds=bounds if method == 'L-BFGS-B' else None,
                options={'maxiter': 5000, 'maxfev': 50000} if method == 'L-BFGS-B' else {'maxiter': 5000, 'maxfev': 50000}
            )
            
            if result.x is not None:
                new_centres = result.x.reshape(N, 3)
                # Check validity
                if np.all(np.isfinite(new_centres)) and np.all(np.abs(new_centres) <= 0.5):
                    new_radius = _compute_radius(new_centres)
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_centres = new_centres
        except Exception:
            continue
    
    return best_centres, best_radius


def _make_initial_packing():
    """Create a good initial packing configuration."""
    # Start with a face-centered cubic-like arrangement
    # or use a grid with perturbation
    
    # Method 1: Cubic grid with small random perturbation
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    rng = np.random.RandomState(42)
    centres += rng.normal(0, 0.01, size=centres.shape)
    
    # Clip to valid range
    centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def construct():
    """Construct centres of N equal spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for safety
    
    # Generate initial packing
    centres = _make_initial_packing()
    
    # Run local optimization
    centres, radius = _local_optimize(centres, time_limit=110.0)
    
    # If we still have time, try a few more restarts with different seeds
    elapsed = time.time() - start_time
    if elapsed < 100.0:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed)
            m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            init_centres = grid[:N].copy()
            init_centres += rng.normal(0, 0.05, size=init_centres.shape)
            init_centres = np.clip(init_centres, -0.49, 0.49)
            
            try:
                new_centres, new_radius = _local_optimize(init_centres, time_limit=5.0)
                if new_radius > radius:
                    radius = new_radius
                    centres = new_centres
            except Exception:
                continue
    
    # Final validation
    if not np.all(np.isfinite(centres)) or np.any(np.abs(centres) > 0.5):
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centres = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return centres
# EVOLVE-BLOCK-END
