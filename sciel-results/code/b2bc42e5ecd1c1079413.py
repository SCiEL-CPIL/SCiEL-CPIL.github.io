"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the largest common radius for the given centres."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Check inside cube [-0.5, 0.5]^3
    boundary = np.min(0.5 - np.abs(centres), axis=1)
    if np.any(boundary < -1e-12):
        return 0.0
    # Check pairwise distances
    # Use vectorized pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r = min(min_pair_dist / 2.0, np.min(boundary))
    return max(0.0, r)


def _initial_grid():
    """Generate an initial grid of N points in the unit cube."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _objective(x, centres_list):
    """
    Objective function for optimization.
    x is a flattened array of centres.
    We maximize the minimum of:
      - half the minimum pairwise distance
      - minimum distance to boundary
    This is equivalent to minimizing the negative of that value.
    """
    centres = x.reshape(-1, 3)
    
    # Boundary constraints: centres must be in [-0.5, 0.5]^3
    # Distance to each face
    dist_to_faces = np.minimum(
        np.minimum(
            np.minimum(
                np.minimum(centres + 0.5, 0.5 - centres[:, 0:1]),
                0.5 - centres[:, 1:2]),
            centres[:, 2:3] + 0.5),
        0.5 - centres[:, 2:3])
    min_boundary = np.min(dist_to_faces)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    r = min(min_pair, min_boundary)
    return -r


def _optimize_from_initial(initial_centres, time_limit=10.0):
    """
    Perform local optimization starting from initial_centres.
    Uses scipy's L-BFGS-B with bounds.
    """
    start_time = time.time()
    x0 = initial_centres.flatten()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10  # Signal timeout
        return _objective(x, None)
    
    best_x = x0.copy()
    best_val = obj(x0)
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if result.success or result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Also try a few more iterations with tighter tolerance if time permits
    elapsed = time.time() - start_time
    if elapsed < time_limit - 1.0:
        try:
            result2 = minimize(
                obj, best_x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 10000}
            )
            if result2.fun < best_val:
                best_x = result2.x
                best_val = result2.fun
        except Exception:
            pass
    
    return best_x.reshape(-1, 3)


def construct():
    """
    Construct the centres of N equal spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    """
    total_start = time.time()
    deadline = total_start + 110.0  # Leave 10s safety margin
    
    # Start with grid
    centres = _initial_grid()
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    # Perform local optimization from the grid
    elapsed = time.time() - total_start
    if elapsed < deadline - 15.0:
        optimized = _optimize_from_initial(centres, time_limit=min(15.0, deadline - time.time() - 5.0))
        r_opt = _compute_radius(optimized)
        if r_opt > best_r:
            best_centres = optimized
            best_r = r_opt
    
    # Try a few random restarts if time permits
    np.random.seed(42)
    attempt = 0
    while time.time() < deadline - 20.0 and attempt < 5:
        # Generate a random starting point with some structure
        # Use a perturbed grid
        perturbation = np.random.normal(0, 0.01, size=(N, 3))
        random_start = best_centres + perturbation
        # Clip to valid range
        random_start = np.clip(random_start, -0.49, 0.49)
        
        remaining = deadline - time.time() - 5.0
        if remaining < 5.0:
            break
        
        optimized = _optimize_from_initial(random_start, time_limit=min(remaining, 15.0))
        r_opt = _compute_radius(optimized)
        if r_opt > best_r:
            best_centres = optimized
            best_r = r_opt
        attempt += 1
    
    # Final polish if time permits
    elapsed = time.time() - total_start
    if elapsed < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            polished = _optimize_from_initial(best_centres, time_limit=min(remaining, 20.0))
            r_polished = _compute_radius(polished)
            if r_polished > best_r:
                best_centres = polished
                best_r = r_polished
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
