"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

# Deadline guard: must finish well before 120s (evaluator kills at 150s)
DEADLINE = time.time() + 100  # 100s safety margin


def _score(positions):
    """Compute the packing radius for given positions."""
    n = len(positions)
    # Wall constraints: distance from each center to the nearest wall
    wall_dists = []
    for i in range(n):
        x, y = positions[i]
        # Square is [-0.5, 0.5]^2
        d = min(x + 0.5, 0.5 - x, y + 0.5, 0.5 - y)
        wall_dists.append(d)
    min_wall = min(wall_dists)
    
    # Pairwise constraints
    min_pair = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_pair:
                min_pair = d
    
    return min(min_wall, min_pair / 2.0)


def _score_vec(positions):
    """Vectorized score computation."""
    positions = np.asarray(positions)
    n = positions.shape[0]
    # Wall distances
    wall_dists = np.min([
        positions[:, 0] + 0.5,
        0.5 - positions[:, 0],
        positions[:, 1] + 0.5,
        0.5 - positions[:, 1]
    ], axis=0)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances (vectorized)
    # Use broadcasting for all pairs
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_wall, min_pair / 2.0)


def _negative_score(positions):
    """Negative score for minimization."""
    pos = positions.reshape(-1, 2)
    return -_score_vec(pos)


def _grid_initial(m=7):
    """Create a grid-based initial guess with some randomization."""
    # Use a 7x7 grid, take 50 points
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take first N points
    return grid[:N].copy()


def _random_initial(n_restarts=10):
    """Generate random initial positions within the square."""
    positions = []
    for _ in range(n_restarts):
        pos = np.random.uniform(-0.45, 0.45, size=(N, 2))
        positions.append(pos)
    return positions


def construct():
    """Main entry point: find optimal packing of 50 circles in unit square."""
    start_time = time.time()
    
    # Start with a good grid-based initial solution
    best_positions = _grid_initial(m=7)
    best_score = _score_vec(best_positions)
    
    # Try several different grid configurations
    for m in [6, 7, 8, 5]:
        if time.time() - start_time > DEADLINE - 20:
            break
        pos = _grid_initial(m=m)
        s = _score_vec(pos)
        if s > best_score:
            best_score = s
            best_positions = pos.copy()
    
    # Local optimization using Nelder-Mead or BFGS on the negative score
    # Use multiple restarts from perturbed versions of the best solution
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Create perturbed initial guess
        if restart == 0:
            x0 = best_positions.flatten()
        else:
            # Add random perturbation
            perturbation_scale = 0.05 * (1 + restart * 0.5)
            x0 = best_positions.flatten() + np.random.uniform(
                -perturbation_scale, perturbation_scale, size=N * 2
            )
            # Clip to valid range
            x0 = np.clip(x0, -0.49, 0.49)
        
        # Optimize using L-BFGS-B with bounds
        bounds = [(-0.49, 0.49)] * (N * 2)
        
        try:
            result = minimize(
                _negative_score,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < -best_score:
                new_pos = result.x.reshape(-1, 2)
                new_score = _score_vec(new_pos)
                if new_score > best_score:
                    best_score = new_score
                    best_positions = new_pos.copy()
        except Exception:
            pass
        
        # Check time
        if time.time() - start_time > DEADLINE - 15:
            break
    
    # Final polish: high-precision local optimization
    if time.time() - start_time < DEADLINE - 5:
        x0 = best_positions.flatten()
        bounds = [(-0.49, 0.49)] * (N * 2)
        try:
            result = minimize(
                _negative_score,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            new_pos = result.x.reshape(-1, 2)
            new_score = _score_vec(new_pos)
            if new_score > best_score:
                best_score = new_score
                best_positions = new_pos.copy()
        except Exception:
            pass
    
    # Verify the result is valid
    final_score = _score_vec(best_positions)
    if final_score <= 0:
        # Fallback to grid
        best_positions = _grid_initial(m=7)
    
    return best_positions
# EVOLVE-BLOCK-END
