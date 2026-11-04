"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the largest common radius for a given set of centres."""
    # Wall constraint: distance from each centre to the nearest wall
    # Cube is [-0.5, 0.5]^3
    wall_dists = np.minimum(0.5 - np.abs(centres), axis=1)
    r_wall = np.min(wall_dists)
    
    # Pairwise constraint: half the minimum distance between any pair
    # Use vectorized computation for efficiency
    n = len(centres)
    # Compute all pairwise distances using broadcasting (memory intensive for large N)
    # For N=100, this is 100x100 = 10000 entries, manageable
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf to avoid self-distance
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    
    r = min(r_wall, r_pair)
    return r


def _objective(centres_flat, target_r):
    """
    Objective function: minimize the violation of having radius >= target_r.
    We want all pairwise distances >= 2*target_r and all wall distances >= target_r.
    """
    centres = centres_flat.reshape(-1, 3)
    # Check if centres are within bounds
    if np.any(np.abs(centres) > 0.5 + 1e-10):
        return 1e6
    
    # Wall violations
    wall_dists = np.minimum(0.5 - np.abs(centres), axis=1)
    wall_viol = np.maximum(0, target_r - wall_dists)
    
    # Pairwise violations
    n = len(centres)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = np.maximum(0, 2 * target_r - dists)
    
    # Sum of violations
    total_viol = np.sum(wall_viol) + np.sum(pair_viol)
    
    # Add a small penalty for centres outside the cube
    out_of_bounds = np.maximum(0, np.abs(centres) - 0.5)
    total_viol += 100 * np.sum(out_of_bounds)
    
    return total_viol


def _grad_objective(centres_flat, target_r):
    """Gradient of the objective function."""
    centres = centres_flat.reshape(-1, 3)
    n = len(centres)
    grad = np.zeros_like(centres)
    
    # Wall gradient
    wall_dists = np.minimum(0.5 - np.abs(centres), axis=1)
    # For each centre, find which wall is closest
    for i in range(n):
        for k in range(3):
            if centres[i, k] >= 0:
                dist_to_wall = 0.5 - centres[i, k]
            else:
                dist_to_wall = 0.5 + centres[i, k]
            if target_r - dist_to_wall > 0:
                # Violation exists, gradient points away from wall
                if centres[i, k] >= 0:
                    grad[i, k] -= 1.0
                else:
                    grad[i, k] += 1.0
    
    # Pairwise gradient
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if 2 * target_r - d > 0:
                # Violation exists, push apart
                if d > 1e-10:
                    direction = diff[i, j] / d
                    grad[i] -= direction
                    grad[j] += direction
    
    return grad.flatten()


def _make_valid(centres):
    """Ensure centres are within the cube."""
    centres = np.clip(centres, -0.5 + 1e-12, 0.5 - 1e-12)
    return centres


def construct():
    """Construct the best packing found within the time budget."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial configuration: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Compute initial radius
    r_current = _compute_radius(centres)
    best_centres = centres.copy()
    best_r = r_current
    
    # Target radius optimization: iteratively increase target radius
    # and minimize violations
    target_r = r_current
    max_iterations = 50
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try to increase target radius
        # Start with a small increment
        target_r = best_r * 1.01  # Try 1% increase
        
        # Optimize to satisfy the new target radius
        x0 = best_centres.flatten()
        
        # Use BFGS optimization
        result = minimize(
            _objective,
            x0,
            args=(target_r,),
            method='BFGS',
            jac=_grad_objective,
            options={'maxiter': 200, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < 1e-6:
            new_centres = _make_valid(result.x.reshape(-1, 3))
            new_r = _compute_radius(new_centres)
            
            if new_r > best_r:
                best_r = new_r
                best_centres = new_centres
        
        # If we can't increase, try a smaller increment
        if _compute_radius(best_centres) < target_r:
            target_r = best_r * 1.001
    
    # Final squeeze: try to polish the best configuration
    if time.time() < deadline - 10:
        # Try multiple random perturbations and optimize
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            # Small random perturbation
            perturbed = best_centres + np.random.normal(0, 0.001, size=best_centres.shape)
            perturbed = _make_valid(perturbed)
            
            result = minimize(
                _objective,
                perturbed.flatten(),
                args=(best_r * 1.005,),
                method='BFGS',
                jac=_grad_objective,
                options={'maxiter': 100, 'gtol': 1e-9}
            )
            
            if result.fun < 1e-6:
                new_centres = _make_valid(result.x.reshape(-1, 3))
                new_r = _compute_radius(new_centres)
                
                if new_r > best_r:
                    best_r = new_r
                    best_centres = new_centres
    
    # Ensure the final result is valid
    best_centres = _make_valid(best_centres)
    
    return best_centres
# EVOLVE-BLOCK-END
