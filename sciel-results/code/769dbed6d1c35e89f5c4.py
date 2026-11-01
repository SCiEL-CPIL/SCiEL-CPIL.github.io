"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree


def _compute_score_fast(centers):
    """
    Compute the radius r = min(min_pair_dist/2, min_wall_dist).
    Uses vectorized NumPy operations for speed.
    """
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Pairwise distances
    # centers shape (N, 2)
    # diff shape (N, N, 2)
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Set diagonal to inf to ignore self-distances
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    min_wall = np.min(0.5 - np.abs(centers))
    
    return min(min_pair, min_wall)


def _simulated_annealing(centers_init, n_iterations, T_start, T_end, base_step, rng):
    """
    Run simulated annealing from an initial configuration.
    Returns (best_centers, best_score).
    """
    centers = centers_init.copy()
    
    current_score = _compute_score_fast(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / n_iterations)
    
    step_size = base_step
    no_improve_count = 0
    max_no_improve = 5000
    
    for it in range(n_iterations):
        T = T_start * (alpha ** it)
        
        # Pick a random center to perturb
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        # Random perturbation
        centers[i] += rng.normal(0, step_size, 2)
        
        # Clip to stay within bounds
        centers[i] = np.clip(centers[i], -0.4999, 0.4999)
        
        new_score = _compute_score_fast(centers)
        
        if new_score >= current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = new_score - current_score
            if delta / T > -100:  # Avoid underflow
                if rng.random() < np.exp(delta / T):
                    current_score = new_score
                    no_improve_count = 0
                else:
                    # Reject: revert
                    centers[i] = old_pos
                    no_improve_count += 1
            else:
                # Reject: revert
                centers[i] = old_pos
                no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > max_no_improve:
            step_size *= 1.2
            no_improve_count = 0
        elif it % 1000 == 0 and it > 0:
            step_size *= 0.95
    
    return best_centers, best_score


def _hexagonal_init(rng, jitter=0.0):
    """
    Generate a hexagonal-like initial configuration with optional jitter.
    """
    centers = np.zeros((N, 2))
    
    # Try to fit 50 circles in a hexagonal pattern
    # Approximate radius r ~ 0.07. Diameter d ~ 0.14.
    # Hex spacing x = d, y = d * sqrt(3)/2 ~ 0.121
    
    d = 0.14
    row_spacing = d * np.sqrt(3) / 2
    
    idx = 0
    y = -0.45
    row = 0
    while idx < N and y <= 0.45:
        x_offset = 0.0 if row % 2 == 0 else d / 2
        x = -0.45 + x_offset
        while x <= 0.45 and idx < N:
            centers[idx] = [x, y]
            idx += 1
            x += d
        y += row_spacing
        row += 1
    
    # Fill remaining if any
    if idx < N:
        remaining = N - idx
        # Place remaining randomly in the center area
        centers[idx:idx+remaining] = rng.uniform(-0.2, 0.2, (remaining, 2))
    
    # Add jitter if specified
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def _random_init(rng):
    """
    Generate a random initial configuration.
    """
    centers = rng.uniform(-0.45, 0.45, (N, 2))
    return centers


def _grid_init(rng, jitter=0.0):
    """
    Generate a grid initial configuration with optional jitter.
    """
    # 7x8 grid = 56, use 50
    rows = 7
    cols = 8
    centers = np.zeros((N, 2))
    
    x_ticks = np.linspace(-0.45, 0.45, cols)
    y_ticks = np.linspace(-0.45, 0.45, rows)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < N:
                centers[idx] = [x_ticks[c], y_ticks[r]]
                idx += 1
            else:
                break
        if idx >= N:
            break
    
    # Add jitter if specified
    if jitter > 0:
        centers += rng.normal(0, jitter, (N, 2))
    
    # Clip to stay within bounds
    centers = np.clip(centers, -0.4999, 0.4999)
    
    return centers


def construct():
    """
    Use simulated annealing with multiple restarts to maximize the minimum
    distance between circles and from walls for 50 equal circles in a unit
    square [-0.5, 0.5]^2.
    
    The score is r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|)).
    We want to maximize r.
    """
    rng = np.random.default_rng(42)
    
    best_overall = None
    best_overall_score = -np.inf
    
    # Multiple restarts with different initializations and parameters
    n_restarts = 4
    
    for restart in range(n_restarts):
        # Choose initialization strategy
        if restart == 0:
            # Hexagonal with small jitter
            centers_init = _hexagonal_init(rng, jitter=0.005)
            n_iters = 100000
            T_start = 0.02
            T_end = 1e-9
            base_step = 0.03
        elif restart == 1:
            # Grid with jitter
            centers_init = _grid_init(rng, jitter=0.01)
            n_iters = 100000
            T_start = 0.02
            T_end = 1e-9
            base_step = 0.035
        elif restart == 2:
            # Hexagonal with more jitter
            centers_init = _hexagonal_init(rng, jitter=0.02)
            n_iters = 100000
            T_start = 0.025
            T_end = 1e-9
            base_step = 0.04
        else:
            # Random init
            centers_init = _random_init(rng)
            n_iters = 100000
            T_start = 0.03
            T_end = 1e-9
            base_step = 0.05
        
        best_centers, best_score = _simulated_annealing(
            centers_init, n_iters, T_start, T_end, base_step, rng
        )
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
    
    return best_overall
# EVOLVE-BLOCK-END
