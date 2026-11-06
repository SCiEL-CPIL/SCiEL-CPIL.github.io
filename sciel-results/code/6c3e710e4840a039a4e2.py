"""Search-based approach for packing 50 equal circles in a unit square."""
import time
import numpy as np
import random

N = 50
SEED = 42
DEADLINE_FRACTION = 0.85

def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    if min_boundary <= 0:
        return 0.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _fast_radius(centers):
    """Fast radius computation without validity checks."""
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_boundary, min_pair)

def _soft_min(values, temperature=0.01):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=float)
    m = np.min(values)
    if not np.isfinite(m):
        return m
    exp_vals = np.exp((values - m) / temperature)
    return m + temperature * np.log(np.sum(exp_vals))

def _smooth_objective(centers, temperature=0.01):
    """Smooth surrogate for the minimum radius objective.
    
    Returns a smooth approximation of min(min_pair_dists/2, min_boundary).
    """
    boundary_dists = 0.5 - np.abs(centers)
    min_boundary = np.min(boundary_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    min_pair = np.min(dists) / 2.0
    
    # Use soft-min to smoothly combine the two constraints
    # We want to maximize the minimum, so we minimize the negative
    # Soft-min of the two values gives a smooth lower bound
    combined = _soft_min(np.array([min_boundary, min_pair]), temperature)
    return combined

def _smooth_gradient(centers, temperature=0.01):
    """Compute gradient of the smooth objective w.r.t. centers.
    
    Returns gradient of -smooth_objective (for maximization via gradient ascent).
    """
    c = centers.copy()
    boundary_dists = 0.5 - np.abs(c)
    
    # Gradient of boundary term
    # d/dx (0.5 - |x|) = -sign(x)
    grad_boundary = -np.sign(c)
    
    # For the pairwise distances, compute gradient of min distance
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Find the minimum distance pair
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_idx
    
    # Gradient of |c_i - c_j| w.r.t. c_i is (c_i - c_j)/|c_i - c_j|
    # Gradient of |c_i - c_j| w.r.t. c_j is -(c_i - c_j)/|c_i - c_j|
    d = diff[i, j]
    dist_ij = dists[i, j]
    if dist_ij > 1e-10:
        unit = d / dist_ij
    else:
        unit = np.array([1.0, 0.0])
    
    grad_pair = np.zeros_like(c)
    grad_pair[i] = unit  # d|ci-cj|/dci
    grad_pair[j] = -unit  # d|ci-cj|/dcj
    
    # The objective is min(boundary, pair_dist/2)
    # For soft-min, we need weighted combination
    min_boundary_val = np.min(boundary_dists)
    min_pair_val = dists[i, j] / 2.0
    
    # Soft-min weights
    vals = np.array([min_boundary_val, min_pair_val])
    m = np.min(vals)
    exp_vals = np.exp((vals - m) / temperature)
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient of soft-min: sum of weights * gradients
    # Gradient of min_boundary w.r.t. centers: need to find which boundary is min
    min_boundary_idx = np.argmin(boundary_dists)
    ci = min_boundary_idx // 2
    axis = min_boundary_idx % 2
    
    grad_boundary_full = np.zeros_like(c)
    grad_boundary_full[ci, axis] = -np.sign(c[ci, axis])
    
    # Gradient of min_pair/2
    grad_pair_half = grad_pair / 2.0
    
    # Combined gradient
    grad = weights[0] * grad_boundary_full + weights[1] * grad_pair_half
    
    # We want to maximize the radius, so we do gradient ascent on radius
    # which is gradient descent on -radius
    return -grad  # Negative because we're maximizing

def _gradient_polish(centers, max_iters=200, lr=0.001, deadline=None):
    """Polish solution using gradient-based optimization on smooth surrogate."""
    c = centers.copy().astype(float)
    best_c = c.copy()
    best_r = _fast_radius(c)
    
    # Adaptive learning rate
    current_lr = lr
    no_improve = 0
    patience = 30
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        grad = _smooth_gradient(c, temperature=0.01)
        
        # Gradient ascent (maximizing radius)
        c_new = c + current_lr * grad
        
        # Clip to bounds
        np.clip(c_new, -0.5 + 1e-8, 0.5 - 1e-8, out=c_new)
        
        # Evaluate
        new_r = _fast_radius(c_new)
        
        if new_r > best_r:
            best_r = new_r
            best_c = c_new.copy()
            c = c_new
            no_improve = 0
        else:
            no_improve += 1
            # Reduce learning rate
            current_lr = max(1e-6, current_lr * 0.9)
            
            if no_improve > patience:
                break
    
    return best_c, best_r

def _initial_grid():
    """Generate initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].astype(float)

def _hexagonal_initializer():
    """Generate hexagonal lattice initial configuration."""
    best_config = None
    best_r = 0
    
    for n_rows in range(5, 12):
        base_per_row = N // n_rows
        extra = N % n_rows
        row_counts = [base_per_row + (1 if i < extra else 0) for i in range(n_rows)]
        
        centers = []
        for row_idx, count in enumerate(row_counts):
            offset = 0.0 if row_idx % 2 == 0 else 0.5 / max(count - 1, 1)
            for col_idx in range(count):
                x = -0.45 + 0.9 * col_idx / max(count - 1, 1) + offset * 0.1
                y = -0.45 + 0.9 * row_idx / max(n_rows - 1, 1)
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r = _fast_radius(centers)
        if r > best_r:
            best_r = r
            best_config = centers
    
    return best_config if best_config is not None else _initial_grid()

def _random_initializer(n_points=N, seed=None):
    """Generate random initial positions within the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.5 + 0.05, 0.5 - 0.05, size=(n_points, 2))

def _force_relaxation(centers, max_iters=500, deadline=None):
    """Repulsive force relaxation to spread circles apart."""
    c = centers.copy().astype(float)
    step_size = 0.01
    min_step = 0.0001
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = np.zeros_like(c)
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(2) * 0.001
                    dist = np.linalg.norm(diff)
                
                if dist < 0.15:
                    force_mag = 0.5 / (dist + 0.01)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        for i in range(N):
            for axis in range(2):
                if c[i, axis] > 0:
                    dist_to_wall = 0.5 - c[i, axis]
                    if dist_to_wall < 0.1:
                        forces[i, axis] -= 0.3 / (dist_to_wall + 0.01)
                else:
                    dist_to_wall = 0.5 + c[i, axis]
                    if dist_to_wall < 0.1:
                        forces[i, axis] += 0.3 / (dist_to_wall + 0.01)
        
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            actual_step = min(step_size, 0.5 / max_force)
            c += actual_step * forces
        
        np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6, out=c)
        step_size = max(min_step, step_size * 0.995)
    
    return c

def _local_search(centers, n_iterations=2000, step_size=0.02, deadline=None):
    """Local search with adaptive step sizes."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    rng = np.random.RandomState(42)
    
    current_step = step_size
    no_improve_count = 0
    max_no_improve = 50
    
    for i in range(n_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = rng.randint(N)
        orig = best_centers[idx].copy()
        
        improved = False
        for attempt in range(3):
            s = current_step * (0.5 ** attempt)
            perturbation = rng.normal(0, s, size=2)
            candidate = orig + perturbation
            
            np.clip(candidate, -0.5 + 1e-8, 0.5 - 1e-8, out=candidate)
            best_centers[idx] = candidate
            
            new_radius = _fast_radius(best_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                improved = True
                break
            else:
                best_centers[idx] = orig
        
        if improved:
            no_improve_count = 0
            current_step = min(0.05, current_step * 1.02)
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                current_step = max(0.001, current_step * 0.8)
                no_improve_count = 0
    
    return best_centers, best_radius

def construct():
    """Main function that searches for the best packing configuration."""
    start_time = time.time()
    total_time_budget = 150.0
    deadline = start_time + total_time_budget * DEADLINE_FRACTION
    
    best_centers = None
    best_radius = 0.0
    
    # Start with multiple initialization strategies
    initializers = [
        _initial_grid(),
        _hexagonal_initializer(),
        _random_initializer(seed=SEED),
        _random_initializer(seed=SEED + 1),
        _random_initializer(seed=SEED + 2),
    ]
    
    for init_centers in initializers:
        r = _compute_radius(init_centers)
        if r > best_radius:
            best_radius = r
            best_centers = init_centers.copy()
    
    # Phase 1: Force relaxation on each initial configuration
    for init_centers in initializers:
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
        
        max_iter = max(50, int(min(500, 500 * remaining_time / 30.0)))
        optimized = _force_relaxation(init_centers, max_iters=max_iter, deadline=deadline)
        
        r = _compute_radius(optimized)
        if r > best_radius:
            best_radius = r
            best_centers = optimized.copy()
    
    # Phase 2: Gradient-based polish on best solution
    if time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 10.0
        max_iters = max(100, int(min(500, 500 * remaining / 20.0)))
        polished, r = _gradient_polish(best_centers, max_iters=max_iters, lr=0.001, deadline=deadline)
        if r > best_radius:
            best_radius = r
            best_centers = polished
    
    # Phase 3: Local search refinement
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 5.0
        n_iters = max(200, int(min(5000, 5000 * remaining / 20.0)))
        best_centers, best_radius = _local_search(best_centers, n_iterations=n_iters, deadline=deadline)
    
    # Phase 4: Final gradient polish with tighter tolerance
    if time.time() < deadline - 2.0:
        remaining = deadline - time.time() - 2.0
        max_iters = max(50, int(min(300, 300 * remaining / 10.0)))
        polished, r = _gradient_polish(best_centers, max_iters=max_iters, lr=0.0005, deadline=deadline)
        if r > best_radius:
            best_radius = r
            best_centers = polished
    
    # Final check
    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        best_centers = _initial_grid()
    
    return best_centers

# EVOLVE-BLOCK-END
