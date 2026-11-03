"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    if n < 2:
        return np.min(0.5 - np.abs(centers))
    
    # Distance to boundaries
    min_boundary = np.min(0.5 - np.abs(centers))
    
    # Minimum pair distance / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _grid_init(n):
    """Generate grid initial configuration."""
    m = int(np.ceil(n ** (1.0/3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]


def _random_init(n, seed=None):
    """Generate random initial configuration inside the cube."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))


def _soft_repulsion_energy(centers, target_r, beta=50.0):
    """
    Smooth surrogate for the nonsmooth min-distance objective.
    Uses log-sum-exp style soft-min for pairwise distances and wall distances.
    Returns an energy that is small when all distances exceed 2*target_r and
    wall distances exceed target_r.
    """
    n = len(centers)
    
    # Pairwise soft-min: we want to maximize the minimum pairwise distance.
    # Equivalently, minimize the "violation" = max(0, 2*target_r - d_ij).
    # Use a smooth approximation: for each pair, penalty = softplus(2*target_r - d_ij) / beta
    # Actually, let's use a different approach: minimize sum of exp(-(d_ij - 2*target_r)*beta)
    # This is large when d_ij < 2*target_r and small when d_ij > 2*target_r.
    
    # For efficiency, compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider pairs that are close (to save computation)
    # But for smoothness, let's use all pairs with a threshold
    # Actually, with n=100, 4950 pairs is fine.
    
    # Soft penalty: for each pair, if d < 2*target_r, add penalty
    # Use: penalty = sum_{i<j} exp(-(d_ij - 2*target_r) * beta) for d_ij < 2*target_r + margin
    # To keep it smooth and bounded, use:
    # penalty_pair = sum_{i<j} max(0, 2*target_r - d_ij)^2 is nonsmooth.
    # Better: use a smooth approximation like:
    # soft_violation = (1/beta) * log(1 + exp(beta * (2*target_r - d_ij)))
    # This is a smooth approximation to max(0, 2*target_r - d_ij).
    
    # For wall distances:
    wall_dists = 0.5 - np.abs(centers)
    # wall_violation_i = max(0, target_r - wall_dist_i)
    # Smooth: (1/beta) * log(1 + exp(beta * (target_r - wall_dist_i)))
    
    # Pairwise soft violation
    # To avoid overflow, clip the argument
    pair_arg = np.clip(beta * (2.0 * target_r - dists), -500, 500)
    pair_soft_viol = np.sum(np.logaddexp(0, pair_arg)) / beta
    
    # Wall soft violation
    wall_arg = np.clip(beta * (target_r - wall_dists), -500, 500)
    wall_soft_viol = np.sum(np.logaddexp(0, wall_arg)) / beta
    
    return pair_soft_viol + wall_soft_viol


def _repulsion_energy(centers, target_r):
    """
    Energy function: penalize distances less than 2*target_r between spheres
    and spheres too close to walls. Uses squared violations for smoothness.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Repulsion: for pairs closer than 2*target_r, add penalty
    close_mask = dists < 2.0 * target_r
    if np.any(close_mask):
        violations = (2.0 * target_r - dists[close_mask])
        pair_energy = np.sum(violations**2)
    else:
        pair_energy = 0.0
    
    # Wall repulsion: spheres should be at least target_r from walls
    wall_dists = 0.5 - np.abs(centers)
    wall_violations = np.maximum(0, target_r - wall_dists)
    wall_energy = np.sum(wall_violations**2)
    
    return pair_energy + wall_energy


def _optimize_from_init(centers_init, target_r, max_iter=2000, tol=1e-10, time_limit=None, start_time=None):
    """
    Optimize from a given initial configuration using L-BFGS-B.
    Returns (optimized_centers, success, final_energy).
    """
    if time_limit is not None and start_time is not None:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return centers_init, False, _repulsion_energy(centers_init, target_r)
    
    def objective(x):
        c = x.reshape(-1, 3)
        return _repulsion_energy(c, target_r)
    
    # Use L-BFGS-B with bounds to keep centers in the cube
    bounds = [(-0.5, 0.5)] * (3 * len(centers_init))
    
    result = minimize(
        objective,
        centers_init.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'gtol': tol, 'disp': False}
    )
    
    return result.x.reshape(-1, 3), result.success, result.fun


def construct():
    """Construct the optimal packing of N spheres in the unit cube."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Generate multiple initializations
    inits = []
    
    # Grid initialization
    grid = _grid_init(N)
    inits.append(grid)
    
    # Random initializations with different seeds
    for seed in range(30):
        if time.time() > deadline - 5.0:
            break
        r_init = _random_init(N, seed=seed)
        inits.append(r_init)
    
    # Also try a few structured perturbations of the grid
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        rng = np.random.RandomState(seed + 1000)
        perturbed_grid = grid + rng.uniform(-0.02, 0.02, size=(N, 3))
        perturbed_grid = np.clip(perturbed_grid, -0.49, 0.49)
        inits.append(perturbed_grid)
    
    # Evaluate all initializations
    scored_inits = []
    for init in inits:
        r_est = _radius(init)
        scored_inits.append((r_est, init))
    
    # Sort by initial radius (best first)
    scored_inits.sort(key=lambda x: -x[0])
    
    # Try to optimize from the top initializations
    for r_est, init in scored_inits:
        if time.time() > deadline - 10.0:
            break
        
        # Use the current best radius as target, but also try slightly higher
        target_r = max(best_radius, r_est)
        
        # If this init is better than current best, use its radius as target
        if r_est > best_radius:
            target_r = r_est
            best_radius = r_est
            best_centers = init.copy()
        
        # Try optimization
        try:
            optimized, success, energy = _optimize_from_init(
                init, target_r, max_iter=1500, tol=1e-9,
                time_limit=deadline - time.time() - 5.0,
                start_time=start_time
            )
            
            # Check if optimization improved the radius
            r_opt = _radius(optimized)
            if r_opt > best_radius:
                best_radius = r_opt
                best_centers = optimized.copy()
        except Exception:
            continue
    
    # Final polish: try to improve the best solution further
    if best_centers is not None and time.time() < deadline - 20.0:
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            
            # Try a higher target radius
            target_r = best_radius * 1.001  # Slightly higher target
            
            try:
                optimized, success, energy = _optimize_from_init(
                    best_centers, target_r, max_iter=1000, tol=1e-10,
                    time_limit=deadline - time.time() - 5.0,
                    start_time=start_time
                )
                r_opt = _radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
            
            # Also try with perturbation
            rng = np.random.RandomState(attempt)
            perturbed = best_centers + rng.randn(N, 3) * 0.005
            perturbed = np.clip(perturbed, -0.499, 0.499)
            
            try:
                optimized, success, energy = _optimize_from_init(
                    perturbed, best_radius, max_iter=500, tol=1e-10,
                    time_limit=deadline - time.time() - 5.0,
                    start_time=start_time
                )
                r_opt = _radius(optimized)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_centers = optimized.copy()
            except Exception:
                continue
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _grid_init(N)
    
    # Final validation: ensure all centers are inside the cube
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    
    return best_centers
# EVOLVE-BLOCK-END
