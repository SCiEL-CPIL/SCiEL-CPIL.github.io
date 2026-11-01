"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree


def _compute_score_fast(centers):
    """
    Compute the radius r = min(min_pair_dist/2, min_wall_dist).
    Uses KDTree for efficient nearest neighbor search.
    """
    n = len(centers)
    if n < 2:
        return 0.5
    
    # Wall distances
    min_wall = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances using KDTree
    # query_kdtree finds the 2 nearest neighbors for each point (including self)
    # We take the 2nd closest (index 1) as the nearest other point
    tree = KDTree(centers)
    # d is shape (N, 2), d[:, 1] is the distance to the nearest neighbor
    d, _ = tree.query(centers, k=2)
    min_pair = np.min(d[:, 1]) / 2.0
    
    return min(min_pair, min_wall)


def _simulated_annealing(centers_init, n_iterations, T_start, T_end, base_step, rng, time_limit=None):
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
    max_no_improve = 2000
    
    import time
    start_time = time.time()
    
    for it in range(n_iterations):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
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
            step_size *= 1.5
            no_improve_count = 0
        elif it % 500 == 0 and it > 0:
            step_size *= 0.98
    
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
    
    import time
    total_time_limit = 110.0 # Leave 10s buffer
    start_global = time.time()
    
    # Multiple restarts with different initializations and parameters
    n_restarts = 6
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_global
        if elapsed > total_time_limit:
            break
            
        # Choose initialization strategy
        if restart % 2 == 0:
            # Hexagonal with small jitter
            jitter = 0.005 + 0.005 * (restart // 2)
            centers_init = _hexagonal_init(rng, jitter=jitter)
            T_start = 0.02 + 0.005 * (restart // 2)
            base_step = 0.03 + 0.005 * (restart // 2)
        else:
            # Grid with jitter
            jitter = 0.01 + 0.005 * (restart // 2)
            centers_init = _grid_init(rng, jitter=jitter)
            T_start = 0.025 + 0.005 * (restart // 2)
            base_step = 0.035 + 0.005 * (restart // 2)
            
        n_iters = 500000 # Higher iterations possible with KDTree
        T_end = 1e-10
        
        # Allocate remaining time for this restart
        time_left = total_time_limit - elapsed
        time_per_restart = time_left / (n_restarts - restart)
        
        best_centers, best_score = _simulated_annealing(
            centers_init, n_iters, T_start, T_end, base_step, rng, time_limit=time_per_restart
        )
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = best_centers
    
    return best_overall
# EVOLVE-BLOCK-END
