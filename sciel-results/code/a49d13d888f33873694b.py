"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: kill after 140s to be safe within 120s limit (with margin)
DEADLINE = 140.0


def _compute_radius(centres):
    """Compute the maximum common radius for a given set of centres."""
    if len(centres) < 2:
        return 0.0
    # Minimum distance to walls
    wall_min = min(0.5 - np.min(centres[:, 0]), 0.5 - np.max(centres[:, 0]),
                   0.5 - np.min(centres[:, 1]), 0.5 - np.max(centres[:, 1]))
    # Minimum pairwise distance / 2
    # Use a more efficient method: compute all pairwise distances
    # For N=200, this is 19900 pairs, manageable
    from scipy.spatial.distance import pdist
    dists = pdist(centres)
    min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
    return min(wall_min, min_pair_dist / 2.0)


def _energy(centres):
    """Energy function to minimize: we want to maximize the minimum radius.
    Use a negative potential that penalizes close pairs and wall violations."""
    # Wall penalty
    x = centres[:, 0]
    y = centres[:, 1]
    # Distance to each wall (should be >= 0 for valid)
    dx_min = x - (-0.5)  # distance to left wall
    dx_max = 0.5 - x     # distance to right wall
    dy_min = y - (-0.5)  # distance to bottom wall
    dy_max = 0.5 - y     # distance to top wall
    
    # We want all these to be large. Penalize when they are small.
    # Use a soft penalty: for any distance d, penalty = max(0, r_target - d)^2
    # But we don't know r_target ahead of time. Instead, use a different approach.
    
    # Alternative: minimize the sum of inverse distances (or squared inverse)
    # to push circles apart and away from walls.
    
    # Pairwise repulsion: sum of 1/d^2 for all pairs
    from scipy.spatial.distance import pdist
    dists = pdist(centres)
    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)
    pair_energy = np.sum(1.0 / (dists * dists))
    
    # Wall repulsion: for each circle, distance to nearest wall
    dist_to_walls = np.minimum(
        np.minimum(dx_min, dx_max),
        np.minimum(dy_min, dy_max)
    )
    dist_to_walls = np.maximum(dist_to_walls, 1e-10)
    wall_energy = np.sum(1.0 / (dist_to_walls * dist_to_walls))
    
    return pair_energy + wall_energy


def _initial_grid():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _random_initial():
    """Create random initial placement inside the square."""
    return (np.random.rand(N, 2) - 0.5) * 0.9  # Keep away from edges initially


def _optimize_repulsion(centres, deadline):
    """Optimize using repulsion-based energy minimization with BFGS/L-BFGS-B."""
    start_time = time.time()
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        if time.time() > deadline:
            return np.inf
        c = x.reshape(-1, 2)
        return _energy(c)
    
    # Try a few iterations with L-BFGS-B
    try:
        result = minimize(
            objective,
            centres.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 5000}
        )
        if result.success or result.fun < np.inf:
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    
    return centres


def _repulsion_step(centres, step_size, deadline):
    """Custom gradient-free repulsion step."""
    # Compute forces
    forces = np.zeros_like(centres)
    
    # Pairwise repulsion
    for i in range(N):
        for j in range(i + 1, N):
            d = centres[i] - centres[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(2) * 1e-10
                dist = np.linalg.norm(d)
            # Repulsive force ~ 1/dist^2
            f = (1.0 / (dist * dist)) * (d / dist)
            forces[i] += f
            forces[j] -= f
    
    # Wall repulsion
    for i in range(N):
        x, y = centres[i]
        # Left wall
        d_left = x - (-0.5)
        if d_left < 0.1:
            forces[i, 0] += 1.0 / max(d_left, 1e-10) ** 2
        # Right wall
        d_right = 0.5 - x
        if d_right < 0.1:
            forces[i, 0] -= 1.0 / max(d_right, 1e-10) ** 2
        # Bottom wall
        d_bottom = y - (-0.5)
        if d_bottom < 0.1:
            forces[i, 1] += 1.0 / max(d_bottom, 1e-10) ** 2
        # Top wall
        d_top = 0.5 - y
        if d_top < 0.1:
            forces[i, 1] -= 1.0 / max(d_top, 1e-10) ** 2
    
    # Update positions
    new_centres = centres + step_size * forces
    # Clamp to square
    new_centres[:, 0] = np.clip(new_centres[:, 0], -0.5, 0.5)
    new_centres[:, 1] = np.clip(new_centres[:, 1], -0.5, 0.5)
    
    return new_centres


def _nelder_mead_optimize(centres, deadline):
    """Optimize using Nelder-Mead on the negative radius objective."""
    start_time = time.time()
    
    def neg_radius(x):
        if time.time() > deadline:
            return 0.0
        c = x.reshape(-1, 2)
        # Clamp to valid region
        c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
        c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
        return -_compute_radius(c)
    
    try:
        result = minimize(
            neg_radius,
            centres.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 100, 'maxfev': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result.fun < np.inf:
            c = result.x.reshape(-1, 2)
            c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
            c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
            return c
    except Exception:
        pass
    
    return centres


def construct():
    """Construct 200 circle centres in the unit square to maximize common radius."""
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_centres = _initial_grid()
    best_radius = _compute_radius(best_centres)
    
    # Strategy: multiple restarts with different initializations, then optimize
    # Use a combination of grid, random, and perturbed grid starts
    
    np.random.seed(42)
    
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Stop if less than 10s left
            break
        
        # Different initialization strategies
        if restart == 0:
            centres = _initial_grid()
        elif restart == 1:
            centres = _random_initial()
        elif restart == 2:
            # Perturbed grid
            centres = _initial_grid() + np.random.randn(N, 2) * 0.01
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        elif restart == 3:
            # Different grid spacing
            m = int(np.ceil(np.sqrt(N)))
            offset = 0.3 / m
            ticks = (np.arange(m) + 0.5) / m - 0.5 + offset
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centres = grid[:N].copy()
            centres[:, 0] = np.clip(centres[:, 0], -0.49, 0.49)
            centres[:, 1] = np.clip(centres[:, 1], -0.49, 0.49)
        else:
            # Another random start
            centres = _random_initial()
        
        # Optimize using repulsion-based method
        if time.time() < deadline - 5:
            centres = _optimize_repulsion(centres, deadline - 5)
        
        # Check radius
        current_radius = _compute_radius(centres)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centres = centres.copy()
        
        # Try Nelder-Mead refinement if time permits
        if time.time() < deadline - 3 and current_radius > best_radius * 0.99:
            refined = _nelder_mead_optimize(centres, deadline - 3)
            refined_radius = _compute_radius(refined)
            if refined_radius > best_radius:
                best_radius = refined_radius
                best_centres = refined.copy()
    
    # Final verification: ensure all centres are strictly inside
    best_centres[:, 0] = np.clip(best_centres[:, 0], -0.5, 0.5)
    best_centres[:, 1] = np.clip(best_centres[:, 1], -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
