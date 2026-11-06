"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _compute_radius(centres):
    """Compute the largest common radius for the given centres."""
    # Ensure centres are within the square [-0.5, 0.5]^2
    # Distance to walls
    wall_dist = np.minimum(
        np.minimum(
            np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
            np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
        ),
        0.5
    )
    min_wall = np.min(wall_dist)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_wall, min_pair / 2.0)


def _smooth_radius_surrogate(centres, beta=50.0):
    """Compute a smooth surrogate for the radius using soft-min.
    
    Uses log-sum-exp trick to create a smooth approximation of the minimum.
    """
    # Distance to walls
    d_left = 0.5 + centres[:, 0]
    d_right = 0.5 - centres[:, 0]
    d_bottom = 0.5 + centres[:, 1]
    d_top = 0.5 - centres[:, 1]
    
    wall_dists = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # Pairwise distances / 2
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_dists = dist / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([wall_dists, pair_dists.flatten()])
    
    # Soft-min via log-sum-exp
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    min_val = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_val))
    soft_min = min_val - np.log(np.sum(exp_vals)) / beta
    
    return soft_min


def _optimize_repulsion(centres, time_limit=100.0):
    """Optimize circle packing using simultaneous repulsion and wall constraints."""
    start_time = time.time()
    n = len(centres)
    centres = centres.copy().astype(np.float64)
    
    # Compute initial radius
    best_r = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Parameters
    learning_rate = 0.002
    decay = 0.998
    repulsion_power = 3  # Moderate repulsion power for stability
    force_cap = 1.0  # Cap the force magnitude to prevent instability
    
    for step in range(100000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(centres)
        
        # Pairwise repulsion
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e-10)  # Avoid division by zero
        
        # Repulsion force: stronger for closer pairs
        repulsion = 1.0 / (dist ** repulsion_power)
        np.fill_diagonal(repulsion, 0)
        
        # Cap repulsion to prevent instability
        repulsion = np.minimum(repulsion, 100.0)
        
        # Force direction
        force_mag = repulsion * learning_rate
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        # Cap total force magnitude
        force_mags = np.linalg.norm(forces, axis=1)
        max_force = force_cap * learning_rate
        for i in range(n):
            if force_mags[i] > max_force:
                forces[i] *= max_force / force_mags[i]
        
        # Wall repulsion: push circles away from walls
        dist_left = centres[:, 0] + 0.5
        dist_right = 0.5 - centres[:, 0]
        dist_bottom = centres[:, 1] + 0.5
        dist_top = 0.5 - centres[:, 1]
        
        # Wall forces (stronger when close to wall)
        wall_strength = 0.3
        forces[:, 0] += wall_strength * learning_rate / (dist_left ** 2 + 1e-8)
        forces[:, 0] -= wall_strength * learning_rate / (dist_right ** 2 + 1e-8)
        forces[:, 1] += wall_strength * learning_rate / (dist_bottom ** 2 + 1e-8)
        forces[:, 1] -= wall_strength * learning_rate / (dist_top ** 2 + 1e-8)
        
        # Cap wall forces too
        wall_force_mags = np.linalg.norm(forces, axis=1)
        for i in range(n):
            if wall_force_mags[i] > max_force:
                forces[i] *= max_force / wall_force_mags[i]
        
        # Update positions
        centres += forces
        
        # Clamp to square
        centres[:, 0] = np.clip(centres[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centres[:, 1] = np.clip(centres[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        # Compute current radius
        current_r = _compute_radius(centres)
        
        if current_r > best_r:
            best_r = current_r
            best_centres = centres.copy()
        
        # Decay learning rate
        learning_rate *= decay
    
    return best_centres, best_r


def _polish_with_scipy(centres, time_limit=20.0):
    """Final polish pass using scipy minimize on smooth surrogate."""
    start_time = time.time()
    centres = centres.copy().astype(np.float64)
    n = len(centres)
    
    # Flatten for scipy
    x0 = centres.flatten()
    
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return 0.0  # Timeout signal
        c = x.reshape(n, 2)
        # Negate because we want to maximize
        return -_smooth_radius_surrogate(c, beta=100.0)
    
    # Use BFGS for smooth optimization
    try:
        result = minimize(
            objective, x0, method='BFGS',
            options={'maxiter': 50, 'gtol': 1e-8, 'maxfun': 100}
        )
        if result.success or (time.time() - start_time < time_limit):
            polished = result.x.reshape(n, 2)
            # Clamp
            polished[:, 0] = np.clip(polished[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
            polished[:, 1] = np.clip(polished[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
            return polished
    except Exception:
        pass
    
    return centres


def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centres = grid[:N].copy()
    
    # Run optimization with random restarts
    best_centres = initial_centres.copy()
    best_r = _compute_radius(initial_centres)
    
    np.random.seed(42)
    num_restarts = 6
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30.0:  # Leave 30s margin for polish
            break
        
        # Create perturbed initial configuration
        if restart == 0:
            centres = initial_centres.copy()
        else:
            # Vary perturbation strength
            perturbation = 0.015 + 0.01 * restart
            centres = initial_centres + np.random.randn(N, 2) * perturbation
            # Clamp to square
            centres[:, 0] = np.clip(centres[:, 0], -0.45, 0.45)
            centres[:, 1] = np.clip(centres[:, 1], -0.45, 0.45)
        
        # Optimize
        result_centres, result_r = _optimize_repulsion(centres, time_limit=15.0)
        
        if result_r > best_r:
            best_r = result_r
            best_centres = result_centres.copy()
    
    # Final polish pass on best configuration
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        polished = _polish_with_scipy(best_centres, time_limit=min(25.0, deadline - elapsed - 5.0))
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best_r = polished_r
            best_centres = polished
    
    return best_centres
