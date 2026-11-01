"""Rewritten program for 50-circle packing in a unit square."""
import numpy as np
import time
import random
import scipy.optimize as opt

N = 50

def _compute_radius(centers):
    """Compute the maximum common radius for a given configuration."""
    # Wall distances: min distance from each center to the four walls
    wall_dists = np.minimum(
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1])
    )
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_pair, min_wall)

def _construct_initial_grid():
    """Create a reasonable initial grid packing."""
    # Use a 7x8 grid (56 positions), take 50
    # Or use a more optimized layout
    # Let's try a hexagonal-like arrangement
    # For now, use a grid and then optimize
    
    m = 7  # 7 columns
    rows_needed = int(np.ceil(N / m))  # 8 rows
    ticks_x = (np.arange(m) + 0.5) / m - 0.5
    ticks_y = (np.arange(rows_needed) + 0.5) / rows_needed - 0.5
    
    centers = []
    for y in ticks_y:
        for x in ticks_x:
            if len(centers) < N:
                centers.append([x, y])
            else:
                break
        if len(centers) >= N:
            break
    
    return np.array(centers[:N])

def _construct_hexagonal():
    """Create a hexagonal packing arrangement."""
    # Hexagonal close packing: rows offset
    # Try to fit 50 circles in a hex pattern
    centers = []
    r = 0.071  # approximate radius
    
    # Hexagonal lattice spacing
    dx = 2 * r  # horizontal spacing
    dy = 2 * r * np.sqrt(3) / 2  # vertical spacing between rows
    
    # Start from center and expand
    # Or just create a grid and perturb
    # Let's use a simple approach: create points in a hex pattern
    
    # Actually, let's just use a grid and let the optimizer do the work
    return _construct_initial_grid()

def _simulated_annealing(centers, time_budget, rng):
    """
    Run simulated annealing to improve the packing.
    """
    start_time = time.time()
    current = centers.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Annealing schedule
        alpha = step / n_steps
        T = T_start * (T_end / T_start) ** alpha
        
        # Pick a random circle to move
        i = rng.randint(0, N - 1)
        
        # Generate a perturbation
        # Scale perturbation by temperature
        scale = T * 2.0
        dx = rng.gauss(0, scale)
        dy = rng.gauss(0, scale)
        
        new_center = current.copy()
        new_center[i] = current[i] + np.array([dx, dy])
        
        # Keep within bounds
        new_center[i, 0] = np.clip(new_center[i, 0], -0.5, 0.5)
        new_center[i, 1] = np.clip(new_center[i, 1], -0.5, 0.5)
        
        new_r = _compute_radius(new_center)
        
        # Accept if better, or with probability exp((new_r - current_r) / T)
        if new_r >= current_r:
            current = new_center
            current_r = new_r
            if new_r > best_r:
                best = new_center.copy()
                best_r = new_r
        else:
            # Probability of accepting worse solution
            delta = current_r - new_r
            if delta < T:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current = new_center
                    current_r = new_r
    
    return best, best_r

def _local_optimization(centers, time_budget):
    """
    Run local optimization using L-BFGS-B to polish the solution.
    """
    start_time = time.time()
    
    def _objective(x):
        """Minimize the negative of the minimum radius (i.e., maximize radius)."""
        c = x.reshape(N, 2)
        # Wall distances
        wall_dists = np.minimum(
            0.5 - np.abs(c[:, 0]),
            0.5 - np.abs(c[:, 1])
        )
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        all_vals = np.concatenate([pair_dists.ravel(), wall_dists])
        # Use soft-min for smoothness
        beta = 50.0
        min_val = np.min(all_vals)
        shifted = -beta * (all_vals - min_val)
        sum_exp = np.sum(np.exp(shifted))
        if sum_exp > 0:
            sm = min_val - (1.0 / beta) * np.log(sum_exp)
        else:
            sm = min_val
        return -sm
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        res = opt.minimize(
            _objective,
            centers.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        x_res = res.x
        r_res = _compute_radius(x_res.reshape(N, 2))
        
        if time.time() - start_time < time_budget:
            return x_res.reshape(N, 2), r_res
    except Exception:
        pass
    
    return centers, _compute_radius(centers)

def construct():
    """
    Main function to construct the optimal packing of 50 circles.
    Returns an array of shape (50, 2) with circle centers.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Start with a grid
    best_centers = _construct_initial_grid()
    best_radius = _compute_radius(best_centers)
    
    # Try multiple restarts with simulated annealing
    n_restarts = 3
    time_per_restart = 35.0  # ~105s total for SA
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Create a perturbed initial configuration
        if restart == 0:
            current = best_centers.copy()
        else:
            # Perturb the best solution found so far
            current = best_centers + rng.normal(0, 0.02, size=(N, 2))
            current = np.clip(current, -0.5, 0.5)
        
        # Run simulated annealing
        sa_centers, sa_radius = _simulated_annealing(current, time_per_restart, rng)
        
        if sa_radius > best_radius:
            best_radius = sa_radius
            best_centers = sa_centers.copy()
    
    # Final polish with local optimization
    if time.time() < deadline - 10.0:
        polish_centers, polish_radius = _local_optimization(best_centers, 10.0)
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_centers = polish_centers.copy()
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    if not np.all(np.isfinite(best_centers)):
        return _construct_initial_grid()
    
    return best_centers
